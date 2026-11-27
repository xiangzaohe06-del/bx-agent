import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from uuid import uuid4

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain_core.language_models.fake_chat_models import FakeListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableLambda
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.errors import GraphRecursionError

from app.agents.orchestrator import init_insurance_agent
from app.agents.schemas import SupervisorDecision, WorkerResult
from app.agents.chitchat import init_chitchat_agent
from app.core.config import settings
from app.modules.chat.service import ChatService
from app.modules.chat.schema import ChatRequest


class SupervisorWorkflowTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.thread_id = uuid4()
        self.config = {"configurable": {"thread_id": str(self.thread_id)}}
        self.supervisor = SimpleNamespace(
            decide=AsyncMock(), respond=AsyncMock(return_value=AIMessage("最终客服回复"))
        )
        self.advisor = SimpleNamespace(ainvoke=AsyncMock(return_value={
            "structured_response": WorkerResult(status="completed", result="产品查询结果", sources=["产品1"]),
            "messages": [AIMessage("内部草稿，不应展示")],
        }))
        self.chitchat = SimpleNamespace(ainvoke=AsyncMock(return_value={
            "messages": [AIMessage("内部闲聊建议")], "chitchat_count": 1,
        }))

    def decision(self, task="查询产品", agent="recommendation_plan"):
        return SupervisorDecision(next_agent=agent, task=task)

    async def graph(self, **kwargs):
        return await init_insurance_agent(
            InMemorySaver(), supervisor=self.supervisor, advisor=self.advisor,
            chitchat=kwargs.get("chitchat", self.chitchat), summary=False,
        )

    async def run_graph(self, graph, text="帮我查询重疾险"):
        return await graph.ainvoke({"messages": [HumanMessage(text)]}, self.config)

    async def test_completed_worker_finishes_without_another_model_decision(self):
        self.supervisor.decide.return_value = self.decision()
        state = await self.run_graph(await self.graph())
        self.supervisor.decide.assert_awaited_once()
        self.supervisor.respond.assert_awaited_once()
        self.assertEqual(state["delegation_count"], 1)
        self.assertEqual(state["worker_results"][0]["sources"], ["产品1"])
        self.assertEqual([message.text for message in state["messages"]], ["帮我查询重疾险", "最终客服回复"])
        self.assertEqual(self.advisor.ainvoke.call_args.args[0]["messages"][-1].type, "system")
        self.chitchat.ainvoke.assert_not_called()

    async def test_missing_user_info_finishes_without_repeated_delegation(self):
        self.supervisor.decide.return_value = self.decision()
        self.advisor.ainvoke.return_value["structured_response"] = WorkerResult(
            status="completed", result="需要了解预算", missing_fields=["预算"]
        )
        state = await self.run_graph(await self.graph())
        self.assertEqual(self.supervisor.decide.await_count, 1)
        self.assertEqual(state["worker_results"][0]["status"], "needs_user_input")
        self.assertIn("补充信息", state["stop_reason"])

    async def test_repeated_task_is_stopped(self):
        self.advisor.ainvoke.return_value["structured_response"] = WorkerResult(
            status="failed", result="本次查询未完成"
        )
        self.supervisor.decide.return_value = self.decision()
        state = await self.run_graph(await self.graph())
        self.advisor.ainvoke.assert_awaited_once()
        self.assertIn("重复任务", state["stop_reason"])

    async def test_finish_without_worker_result_still_requires_business_check(self):
        self.supervisor.decide.return_value = self.decision(agent="FINISH")
        state = await self.run_graph(await self.graph())
        self.advisor.ainvoke.assert_awaited_once()
        self.assertEqual(state["worker_results"][0]["status"], "completed")

    async def test_different_tasks_still_obey_delegation_limit(self):
        self.advisor.ainvoke.return_value["structured_response"] = WorkerResult(
            status="failed", result="本次查询未完成"
        )
        self.supervisor.decide.side_effect = [self.decision(f"补查任务{i}") for i in range(10)]
        state = await self.run_graph(await self.graph())
        self.assertEqual(self.advisor.ainvoke.await_count, settings.supervisor.max_delegations)
        self.assertIn("最大委派次数", state["stop_reason"])

    async def test_failed_worker_does_not_leak_exception_details(self):
        self.supervisor.decide.side_effect = [self.decision(), self.decision(agent="FINISH")]
        self.advisor.ainvoke.side_effect = TimeoutError("private-api-response")
        state = await self.run_graph(await self.graph())
        self.assertEqual(state["worker_results"][0]["status"], "failed")
        self.assertNotIn("private-api-response", str(state))

    async def test_invalid_worker_result_becomes_failure(self):
        self.supervisor.decide.side_effect = [self.decision(), self.decision(agent="FINISH")]
        self.advisor.ainvoke.return_value = {"structured_response": {"status": "invented"}}
        state = await self.run_graph(await self.graph())
        self.assertEqual(state["worker_results"][0]["status"], "failed")

    async def test_greeting_bypasses_models(self):
        state = await self.run_graph(await self.graph(), "你好")
        self.supervisor.decide.assert_not_called()
        self.supervisor.respond.assert_not_called()
        self.assertIn("您好", state["messages"][-1].text)

    async def test_unavailable_handoff_never_claims_transfer_success(self):
        self.supervisor.decide.return_value = self.decision(agent="human_handoff")
        state = await self.run_graph(await self.graph(), "转人工")
        self.assertIn("尚未接入", state["messages"][-1].text)
        self.assertNotIn("已转人工", state["messages"][-1].text)

    async def test_supervisor_and_final_reply_errors_have_safe_fallbacks(self):
        self.supervisor.decide.side_effect = TimeoutError()
        state = await self.run_graph(await self.graph())
        self.assertIn("稍后重试", state["messages"][-1].text)
        self.supervisor.decide.side_effect = None
        self.supervisor.decide.return_value = self.decision(agent="FINISH")
        self.supervisor.respond.return_value = AIMessage("")
        state = await self.run_graph(await self.graph())
        self.assertIn("聊天记录已保留", state["messages"][-1].text)

    async def test_new_turn_clears_previous_tasks_but_preserves_history(self):
        self.supervisor.decide.side_effect = [
            self.decision("第一个任务"), self.decision("第二个任务"),
        ]
        self.supervisor.respond.side_effect = [AIMessage("第一轮回复"), AIMessage("第二轮回复")]
        graph = await self.graph()
        await self.run_graph(graph)
        state = await self.run_graph(graph, "继续比较")
        self.assertEqual(len(state["worker_results"]), 1)
        self.assertEqual(state["worker_results"][0]["task"], "第二个任务")
        self.assertEqual(state["delegation_count"], 1)
        self.assertEqual(len(state["messages"]), 4)
        self.assertEqual((await graph.aget_state({"configurable": {"thread_id": "other"}})).values, {})

    async def test_chitchat_counter_is_preserved_across_stateless_worker_calls(self):
        model = Mock()
        model.with_config.return_value = model
        model.ainvoke = AsyncMock(return_value=AIMessage("普通闲聊建议"))
        with patch("app.agents.chitchat.create_chat_model", return_value=model):
            chitchat = init_chitchat_agent()
        self.supervisor.decide.side_effect = [
            self.decision("处理闲聊", "chitchat") for _ in range(3)
        ]
        graph = await self.graph(chitchat=chitchat)
        for _ in range(3):
            state = await self.run_graph(graph, "讲个小故事")
        self.assertEqual(model.ainvoke.await_count, 2)
        self.assertEqual(state["chitchat_count"], 2)
        self.assertIn("主要协助", state["worker_results"][-1]["result"])

    async def test_sse_streams_only_final_response_and_archive_has_no_worker_draft(self):
        self.supervisor.decide.side_effect = [self.decision(), self.decision(agent="FINISH")]
        model = FakeListChatModel(responses=["最终客服回复"])

        async def respond(state):
            return await model.ainvoke(state["messages"])

        async def worker(state):
            draft = await FakeListChatModel(responses=["内部草稿不能展示"]).ainvoke(state["messages"])
            return {"structured_response": WorkerResult(status="completed", result="已查到产品"), "messages": [draft]}

        self.supervisor.respond.side_effect = respond
        self.advisor = RunnableLambda(worker)
        service = ChatService(None, await self.graph())
        service.repository.required_owner = AsyncMock(return_value=True)
        with patch("app.modules.chat.service.archive_messages", new_callable=AsyncMock) as archive:
            events = [event async for event in service.chat_stream(
                1, ChatRequest(thread_id=self.thread_id, message="查询重疾险")
            )]
        visible = [event.data for event in events if event.event == "message"]
        self.assertEqual("".join(visible), "最终客服回复")
        self.assertGreater(len(visible), 1)
        self.assertEqual(events[-1].event, "done")
        saved_messages = archive.call_args.args[1]
        self.assertEqual([message.text for message in saved_messages], ["查询重疾险", "最终客服回复"])

    async def test_direct_response_skips_workers_and_keeps_original_user_request(self):
        self.supervisor.decide.return_value = self.decision(
            task="不应扩展为查询产品", agent="direct_response"
        )
        request = "我是30岁男性，预算3000元。只告诉我下一步要确认什么。"
        state = await self.run_graph(await self.graph(), request)
        self.supervisor.decide.assert_awaited_once()
        self.supervisor.respond.assert_awaited_once()
        self.advisor.ainvoke.assert_not_called()
        self.chitchat.ainvoke.assert_not_called()
        self.assertEqual(state["active_workflow"], "direct_response")
        self.assertEqual(state["current_task"], request)
        self.assertEqual(state["delegation_count"], 0)
        self.assertEqual(state["worker_results"], [])
        self.assertEqual([message.text for message in state["messages"]], [request, "最终客服回复"])

    async def test_direct_budget_update_preserves_history_and_next_clause_request_can_delegate(self):
        # 每轮生成独立消息，避免复用Mock返回对象的同一message id覆盖上一轮回复。
        self.supervisor.respond.side_effect = [AIMessage("首轮确认"), AIMessage("预算已更新"), AIMessage("条款回复")]
        self.supervisor.decide.side_effect = [
            self.decision(agent="direct_response"), self.decision(agent="direct_response"),
            self.decision(task="查询达尔文12号等待期"),
        ]
        graph = await self.graph()
        await self.run_graph(graph, "我是30岁男性，为自己了解重疾险，年预算5000元。先记录，不推荐。")
        state = await self.run_graph(graph, "预算改成3000元，其他不变。")
        self.assertEqual(len(state["messages"]), 4)
        self.assertIn("5000", self.supervisor.respond.call_args.args[0]["messages"][0].text)
        self.assertIn("3000", self.supervisor.respond.call_args.args[0]["messages"][-1].text)
        self.advisor.ainvoke.assert_not_called()
        state = await self.run_graph(graph, "同时请查询达尔文12号的等待期和条款依据。")
        self.advisor.ainvoke.assert_awaited_once()
        self.assertEqual(state["active_workflow"], "recommendation_plan")
        self.assertEqual(state["delegation_count"], 1)
        self.assertEqual(len(state["messages"]), 6)

    async def test_recursion_error_stops_without_retry_or_exposing_error_details(self):
        self.supervisor.decide.return_value = self.decision()
        self.advisor.ainvoke.side_effect = GraphRecursionError("private-api-response")
        state = await self.run_graph(await self.graph())
        self.advisor.ainvoke.assert_awaited_once()
        self.supervisor.decide.assert_awaited_once()
        self.supervisor.respond.assert_awaited_once()
        self.assertEqual(state["delegation_count"], 1)
        self.assertEqual(state["worker_results"][0]["status"], "failed")
        self.assertFalse(state["worker_results"][0]["retryable"])
        self.assertIn("递归上限", state["stop_reason"])
        self.assertNotIn("private-api-response", str(state))

    async def test_recoverable_failure_can_retry_then_finish_after_success(self):
        self.supervisor.decide.side_effect = [self.decision(), self.decision("换用章节编号补查")]
        self.advisor.ainvoke.side_effect = [TimeoutError("private-api-response"), {
            "structured_response": WorkerResult(status="completed", result="查询成功", sources=["2.3.1"]),
        }]
        state = await self.run_graph(await self.graph())
        self.assertEqual(self.supervisor.decide.await_count, 2)
        self.assertEqual(self.advisor.ainvoke.await_count, 2)
        self.assertEqual(state["worker_results"][-1]["status"], "completed")
        self.supervisor.respond.assert_awaited_once()


if __name__ == '__main__':
    unittest.main(verbosity=2)
