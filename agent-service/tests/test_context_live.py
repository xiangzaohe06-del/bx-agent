import asyncio
import json
import os
import pathlib
import sys
import unittest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))


@unittest.skipUnless(os.getenv("RUN_CONTEXT_LIVE_TESTS") == "1", "需显式开启百炼真实调用测试，会消耗token")
class ContextLiveTests(unittest.IsolatedAsyncioTestCase):
    loop_factory = asyncio.SelectorEventLoop if sys.platform == "win32" else None

    async def test_real_summary_chat_and_history(self):
        import httpx
        from langchain_core.messages import HumanMessage, AIMessage
        from app.core.config import settings
        from app.main import app
        from app.agents.context import is_internal_message

        # 只降低本进程测试阈值，不修改.env，也不需要构造昂贵的超长对话。
        original_trigger = settings.context.summary_trigger_tokens
        original_keep = settings.context.summary_keep_tokens
        settings.context.summary_trigger_tokens = 60
        settings.context.summary_keep_tokens = 25
        try:
            async with app.router.lifespan_context(app):
                async with httpx.AsyncClient(
                    transport=httpx.ASGITransport(app=app), base_url="http://test",
                    headers={"x-user-id": "1"}, timeout=120,
                ) as client:
                    response = await client.post("/api/v1/chat-threads", json={"title": "百炼摘要验证（自动清理）"})
                    response.raise_for_status()
                    thread_id = response.json()["id"]
                    config = {"configurable": {"thread_id": thread_id}}
                    try:
                        await app.state.agent.aupdate_state(config, {"messages": [
                            HumanMessage("我30岁，预算每年3000元，想买重疾险。", id="live1"),
                            AIMessage("好的，您还可以告诉我保障期限。", id="live2"),
                            HumanMessage("预算修改为每年5000元，请以这个新预算为准。", id="live3"),
                            AIMessage("已记录预算变化，请确认保障到多少岁。", id="live4"),
                            HumanMessage("希望保障至70岁，现在还没决定具体产品。", id="live5"),
                            AIMessage("已记录您的保障期限。", id="live6"),
                        ]})
                        response = await client.post("/api/v1/chat", json={
                            "thread_id": thread_id,
                            "message": "继续我的重疾险咨询，请只确认我最新的年预算和保障到多少岁，暂时不要推荐产品或查询条款。",
                        })
                        response.raise_for_status()
                        self.assertIn("event: done", response.text)
                        self.assertNotIn("以下为较早对话的摘要", response.text)
                        reply = "".join(
                            data for line in response.text.splitlines() if line.startswith("data: ")
                            if isinstance(data := json.loads(line[6:]), str) and data != "END"
                        )
                        self.assertIn("5000", reply)
                        self.assertIn("70", reply)
                        snapshot = await app.state.agent.aget_state(config)
                        self.assertGreaterEqual(snapshot.values["delegation_count"], 1)
                        self.assertEqual(snapshot.values["worker_results"][-1]["status"], "completed")
                        self.assertTrue(is_internal_message(snapshot.values["messages"][0]))
                        self.assertIn("5000", snapshot.values["messages"][0].text)
                        self.assertIn("70", snapshot.values["messages"][0].text)
                        history = await client.get(f"/api/v1/chat-threads/{thread_id}/messages")
                        history.raise_for_status()
                        messages = history.json()["messages"]
                        self.assertGreaterEqual(len(messages), 8)
                        self.assertEqual(messages[0]["content"], "我30岁，预算每年3000元，想买重疾险。")
                        print("百炼Supervisor委派、结构化子Agent结果、摘要、SSE与完整历史：通过")

                        # 再走一次真实条款工具，确保新结构化输出仍能调用已有RAG能力。
                        response = await client.post("/api/v1/chat", json={
                            "thread_id": thread_id,
                            "message": "请查询产品ID为2的达尔文12号正式条款，告诉我重大疾病等待期是多少天，并给出查询到的条款依据。",
                        })
                        response.raise_for_status()
                        self.assertIn("event: done", response.text)
                        clause_reply = "".join(
                            data for line in response.text.splitlines() if line.startswith("data: ")
                            if isinstance(data := json.loads(line[6:]), str) and data != "END"
                        )
                        self.assertIn("180", clause_reply)
                        snapshot = await app.state.agent.aget_state(config)
                        self.assertEqual(snapshot.values["worker_results"][-1]["status"], "completed")
                        self.assertTrue(snapshot.values["worker_results"][-1]["sources"])
                        print("Supervisor真实RAG查询、条款依据与最终回复：通过")
                    finally:
                        # 无论断言是否成功，都通过业务接口清理本次测试会话。
                        deleted = await client.delete(f"/api/v1/chat-threads/{thread_id}")
                        deleted.raise_for_status()
        finally:
            settings.context.summary_trigger_tokens = original_trigger
            settings.context.summary_keep_tokens = original_keep


if __name__ == '__main__':
    unittest.main(verbosity=2)
