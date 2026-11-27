import asyncio
import os
import pathlib
import sys
import unittest
from types import SimpleNamespace
from uuid import uuid4

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import select, func

from app.agents.context import ConversationSummaryMiddleware, is_internal_message
from app.agents.schemas import SupervisorDecision, WorkerResult
from app.infra.checkpointer import init_checkpointer, close_checkpointer
from app.infra.database import AsyncSessionFactory, close_database
from app.modules.chat.schema import ChatRequest
from app.modules.chat_thread.history import init_chat_history, archive_messages
from app.modules.chat_thread.models import ChatThread, ChatMessage
from app.modules.chat_thread.service import ChatThreadService


@unittest.skipUnless(os.getenv("RUN_CONTEXT_DB_TESTS") == "1", "需显式开启本机数据库集成测试")
class ContextPersistenceTests(unittest.IsolatedAsyncioTestCase):
    # 与项目Windows启动方式保持一致，psycopg不支持默认的Proactor事件循环。
    loop_factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None

    async def test_archive_summary_restore_stream_and_delete(self):
        # 数据库、父子图和业务服务使用真实实现；模型替换为固定结果，不消耗云端token。
        from app.agents.orchestrator import init_insurance_agent
        from app.modules.chat.service import ChatService

        saver = await init_checkpointer()
        thread_id = uuid4()
        config = {"configurable": {"thread_id": str(thread_id)}}
        middleware = ConversationSummaryMiddleware(
            model=FakeMessagesListChatModel(responses=[AIMessage(
                "用户已将预算从3000元改为5000元，保障至70岁，尚未选择产品。",
                response_metadata={"finish_reason": "stop"},
            )]),
            trigger=("messages", 5), keep=("messages", 2), trim_tokens_to_summarize=None,
        )

        async def decide(state):
            if state.get("worker_results"):
                return SupervisorDecision(next_agent="FINISH")
            return SupervisorDecision(next_agent="recommendation_plan", task="确认最新预算和保障期限")

        async def respond(state):
            return AIMessage("已确认，预算5000元，保障至70岁。")

        async def invoke_worker(state, config):
            return {
                "structured_response": WorkerResult(status="completed", result="最新预算5000元，保障至70岁"),
                "messages": [AIMessage("内部任务草稿，不应归档")],
            }

        supervisor = SimpleNamespace(decide=decide, respond=respond)
        worker = SimpleNamespace(ainvoke=invoke_worker)

        async def init_graph():
            return await init_insurance_agent(
                saver, supervisor=supervisor, advisor=worker, chitchat=worker, summary=middleware
            )

        try:
            agent = await init_graph()
            async with AsyncSessionFactory.begin() as session:
                session.add(ChatThread(id=thread_id, user_id=1, title="摘要集成测试（自动清理）"))
            old_messages = [
                HumanMessage("预算3000元", id="old1"),
                AIMessage("请确认需求", id="old2"),
                HumanMessage("预算提高到5000元", id="old3"),
                AIMessage("还需要保障期限", id="old4"),
                HumanMessage("保障至70岁", id="old5"),
                AIMessage("已记录", id="old6"),
            ]
            await agent.aupdate_state(config, {"messages": old_messages})

            # 模拟启动回填，重复执行不能产生重复历史。
            await init_chat_history(agent)
            await archive_messages(thread_id, old_messages)
            async with AsyncSessionFactory() as session:
                history = await ChatThreadService(session, agent).get_chat_history(1, str(thread_id))
                self.assertEqual(len(history.messages), 6)

            async with AsyncSessionFactory() as session:
                events = [event async for event in ChatService(session, agent).chat_stream(
                    1, ChatRequest(thread_id=thread_id, message="确认预算和保障期限")
                )]
            self.assertTrue(any(event.event == "message" for event in events))
            self.assertEqual(events[-1].event, "done")
            self.assertNotIn("用户已将预算", "".join(str(event.data) for event in events))
            self.assertEqual("".join(event.data for event in events if event.event == "message"),
                             "已确认，预算5000元，保障至70岁。")

            snapshot = await agent.aget_state(config)
            self.assertTrue(is_internal_message(snapshot.values["messages"][0]))
            self.assertLess(len(snapshot.values["messages"]), 8)
            async with AsyncSessionFactory() as session:
                service = ChatThreadService(session, agent)
                history = await service.get_chat_history(1, str(thread_id))
                self.assertEqual(len(history.messages), 8)
                self.assertEqual(history.messages[0].content, "预算3000元")
                self.assertEqual(history.messages[2].content, "预算提高到5000元")
                self.assertEqual(history.messages[-1].content, "已确认，预算5000元，保障至70岁。")

            # 重建Graph后仍从PostgreSQL恢复同一份压缩状态。
            restored_agent = await init_graph()
            restored = await restored_agent.aget_state(config)
            self.assertEqual(restored.values["messages"], snapshot.values["messages"])
            self.assertEqual((await restored_agent.aget_state(
                {"configurable": {"thread_id": str(uuid4())}}
            )).values, {})

            # 切到固定寒暄再返回保险咨询，归档和压缩上下文都应继续正常工作。
            async with AsyncSessionFactory() as session:
                _ = [event async for event in ChatService(session, restored_agent).chat_stream(
                    1, ChatRequest(thread_id=thread_id, message="你好")
                )]
            async with AsyncSessionFactory() as session:
                _ = [event async for event in ChatService(session, restored_agent).chat_stream(
                    1, ChatRequest(thread_id=thread_id, message="继续刚才的预算和保障期限")
                )]
            async with AsyncSessionFactory() as session:
                history = await ChatThreadService(session, restored_agent).get_chat_history(1, str(thread_id))
                self.assertEqual(len(history.messages), 12)
                self.assertIn("您好", history.messages[-3].content)
                self.assertNotIn("用户已将预算", "".join(message.content for message in history.messages))
                self.assertNotIn("内部任务草稿", "".join(message.content for message in history.messages))

            # 走项目原有删除接口，归档消息和检查点必须一起清理。
            async with AsyncSessionFactory() as session:
                await ChatThreadService(session, restored_agent).remove(1, str(thread_id))
            async with AsyncSessionFactory() as session:
                count = await session.scalar(select(func.count()).select_from(ChatMessage).where(
                    ChatMessage.thread_id == thread_id
                ))
                self.assertEqual(count, 0)
            self.assertEqual((await restored_agent.aget_state(config)).values, {})
        finally:
            # 仅清理本测试生成的会话，不触碰用户会话或知识库。
            async with AsyncSessionFactory.begin() as session:
                thread = await session.get(ChatThread, thread_id)
                if thread:
                    await session.delete(thread)
            await saver.adelete_thread(str(thread_id))
            await close_checkpointer()
            await close_database()


if __name__ == '__main__':
    unittest.main(verbosity=2)
