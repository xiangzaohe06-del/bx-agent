import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch
from uuid import uuid4

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.checkpoint.memory import InMemorySaver

from app.agents.context import ConversationSummaryMiddleware, is_internal_message
from app.agents.orchestrator import init_insurance_agent
from app.agents.schemas import SupervisorDecision, WorkerResult
from app.modules.chat_thread.history import message_records


class ConversationContextTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.thread_id = str(uuid4())
        self.runtime = SimpleNamespace(execution_info=SimpleNamespace(thread_id=self.thread_id))
        self.messages = [
            HumanMessage("预算3000元", id="u1"),
            AIMessage("可以比较产品A", id="a1"),
            HumanMessage("预算提高到5000元", id="u2"),
            AIMessage("请确认保障期限", id="a2"),
            HumanMessage("保到70岁", id="u3"),
        ]
        self.archive = AsyncMock()
        self.archive_patch = patch("app.agents.context.archive_messages", self.archive)
        self.archive_patch.start()
        self.addCleanup(self.archive_patch.stop)

    def middleware(self, response=None):
        response = response if response is not None else AIMessage(
            "预算现为5000元，保至70岁", response_metadata={"finish_reason": "stop"}
        )
        return ConversationSummaryMiddleware(
            FakeMessagesListChatModel(responses=[response]),
            trigger=("messages", 4), keep=("messages", 2), trim_tokens_to_summarize=None,
        )

    async def test_short_conversation_only_archives(self):
        middleware = self.middleware()
        middleware._acreate_summary = AsyncMock()
        update = await middleware.abefore_model({"messages": self.messages[:2]}, self.runtime)
        self.assertIsNone(update)
        middleware._acreate_summary.assert_not_called()
        self.archive.assert_awaited_once()

    async def test_summary_keeps_recent_messages_and_marks_internal_context(self):
        update = await self.middleware().abefore_model({"messages": self.messages}, self.runtime)
        self.assertEqual(update["messages"][0].type, "remove")
        self.assertTrue(is_internal_message(update["messages"][1]))
        self.assertEqual(update["messages"][-2:], self.messages[-2:])
        self.assertIn("5000", update["messages"][1].text)
        self.archive.assert_awaited_once_with(self.thread_id, self.messages)

    async def test_failed_empty_or_truncated_summary_never_replaces_history(self):
        for response in [
            AIMessage("部分摘要", response_metadata={"finish_reason": "length"}),
            AIMessage("", response_metadata={"finish_reason": "stop"}),
            AIMessage("原因缺失"),
        ]:
            with self.subTest(response=response):
                self.assertIsNone(await self.middleware(response).abefore_model(
                    {"messages": self.messages}, self.runtime
                ))
        middleware = self.middleware()
        middleware._acreate_summary = AsyncMock(side_effect=TimeoutError())
        self.assertIsNone(await middleware.abefore_model({"messages": self.messages}, self.runtime))
        self.assertEqual(len(self.messages), 5)

    async def test_archive_failure_prevents_compression(self):
        self.archive.side_effect = RuntimeError("归档失败")
        middleware = self.middleware()
        middleware._acreate_summary = AsyncMock()
        with self.assertRaises(RuntimeError):
            await middleware.abefore_model({"messages": self.messages}, self.runtime)
        middleware._acreate_summary.assert_not_called()

    async def test_tool_call_and_result_are_kept_together(self):
        messages = self.messages[:2] + [
            AIMessage("", id="call", tool_calls=[{"id": "t1", "name": "lookup", "args": {}}]),
            ToolMessage("条款原文", tool_call_id="t1", id="result"),
            AIMessage("条款解释", id="answer"),
        ]
        update = await self.middleware().abefore_model({"messages": messages}, self.runtime)
        kept = update["messages"][2:]
        self.assertEqual([message.id for message in kept], ["call", "result", "answer"])

    def test_archive_excludes_internal_context_and_preserves_original_text(self):
        summary = self.middleware()._build_new_messages("内部摘要")[0]
        messages = self.messages + [summary, ToolMessage("内部工具结果", tool_call_id="t1")]
        records = message_records(self.thread_id, messages)
        self.assertEqual(len(records), 5)
        self.assertEqual(records[0]["content"], "预算3000元")
        self.assertEqual(records[2]["content"], "预算提高到5000元")
        self.assertEqual(records, message_records(self.thread_id, messages))

    def test_internal_event_detection(self):
        self.assertTrue(is_internal_message({"delta": {"text": "内部摘要"}}, {"context_summary": True}))
        self.assertTrue(is_internal_message(AIMessage("内部摘要"), {"lc_source": "summarization"}))
        self.assertFalse(is_internal_message(AIMessage("正常回复"), {}))

    async def test_worker_summary_does_not_archive_internal_drafts(self):
        middleware = self.middleware()
        middleware.archive_history = False
        update = await middleware.abefore_model({"messages": self.messages}, SimpleNamespace())
        await middleware.aafter_model({"messages": self.messages}, SimpleNamespace())
        self.assertTrue(is_internal_message(update["messages"][1]))
        self.archive.assert_not_called()

    async def test_nested_graph_restores_compacted_state_and_isolates_threads(self):
        # 使用真实Supervisor主图和摘要，只替换模型及归档，验证公开上下文的恢复。
        middleware = self.middleware()
        supervisor = SimpleNamespace(
            decide=AsyncMock(side_effect=[SupervisorDecision(next_agent="FINISH")] * 4),
            respond=AsyncMock(side_effect=[AIMessage("正常回复"), AIMessage("正常回复")]),
        )
        advisor = SimpleNamespace(ainvoke=AsyncMock(return_value={
            "structured_response": WorkerResult(status="completed", result="任务已完成")
        }))
        saver = InMemorySaver()
        graph = await init_insurance_agent(
            saver, supervisor=supervisor, advisor=advisor, chitchat=advisor, summary=middleware
        )
        config = {"configurable": {"thread_id": self.thread_id}}
        result = await graph.ainvoke({"messages": self.messages}, config)
        self.assertEqual(len(result["messages"]), 4)
        self.assertTrue(is_internal_message(result["messages"][0]))
        self.assertNotIn("u1", [message.id for message in result["messages"]])
        self.assertEqual((await graph.aget_state(config)).values["messages"], result["messages"])
        self.assertEqual((await graph.aget_state({"configurable": {"thread_id": "other"}})).values, {})

        # 重新编译模拟恢复，历史不能重新膨胀，内部摘要也不能成为用户可见输出。
        restored = await init_insurance_agent(
            saver, supervisor=supervisor, advisor=advisor, chitchat=advisor, summary=middleware
        )
        stream = await restored.astream_events(
            {"messages": [HumanMessage("继续比较", id="u4")]}, config, version="v3"
        )
        visible = []
        async for event in stream:
            if event["method"] == "messages":
                message, metadata = event["params"]["data"]
                if not is_internal_message(message, metadata):
                    visible.append(str(message))
        self.assertNotIn("预算现为5000元", "".join(visible))
        final_state = (await restored.aget_state(config)).values["messages"]
        self.assertTrue(is_internal_message(final_state[0]))
        self.assertEqual(final_state[-1].text, "正常回复")


if __name__ == '__main__':
    unittest.main(verbosity=2)
