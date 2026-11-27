import asyncio
import hashlib
import json
from contextlib import asynccontextmanager
from time import perf_counter
from uuid import uuid4

from langchain_core.callbacks import BaseCallbackHandler
from langsmith import get_current_run_tree, tracing_context

from app.core.config import APP_ROOT, settings

# 使用独立的测试用户，避免测评会话出现在正常用户的聊天列表中。
EVALUATION_USER_ID = -9000001


class EvaluationTraceCollector(BaseCallbackHandler):
    """采集真实工具调用与聊天模型用量，原生向量SDK用量不混入聊天token统计。"""
    run_inline = True

    def __init__(self):
        self.tools = {}
        self.model_starts = set()
        self.model_ends = set()
        self.input_tokens = 0
        self.output_tokens = 0
        self.known_usage_calls = 0

    def on_chat_model_start(self, serialized, messages, *, run_id, **kwargs):
        self.model_starts.add(str(run_id))

    def on_llm_end(self, response, *, run_id, **kwargs):
        if str(run_id) in self.model_ends:
            return
        self.model_ends.add(str(run_id))
        usage = None
        for generation in response.generations:
            if generation:
                usage = getattr(getattr(generation[0], "message", None), "usage_metadata", None)
                if usage:
                    break
        if not usage:
            raw = (response.llm_output or {}).get("token_usage", {})
            if "prompt_tokens" in raw and "completion_tokens" in raw:
                usage = {"input_tokens": raw["prompt_tokens"], "output_tokens": raw["completion_tokens"]}
        if usage and isinstance(usage.get("input_tokens"), int) and isinstance(usage.get("output_tokens"), int):
            self.input_tokens += usage["input_tokens"]
            self.output_tokens += usage["output_tokens"]
            self.known_usage_calls += 1

    def on_tool_start(self, serialized, input_str, *, run_id, **kwargs):
        # LangChain会额外传入原始inputs；input_str可能是Python字典文本，不能只按JSON解析。
        arguments = kwargs.get("inputs")
        if not isinstance(arguments, dict):
            try:
                arguments = json.loads(input_str)
            except (ValueError, TypeError):
                arguments = {}
        self.tools[str(run_id)] = {
            "name": (serialized or {}).get("name", ""),
            "arguments": arguments if isinstance(arguments, dict) else {},
            "output": "", "error": False,
        }

    def on_tool_end(self, output, *, run_id, **kwargs):
        record = self.tools.get(str(run_id))
        if record is not None:
            content = getattr(output, "content", output)
            text = content if isinstance(content, str) else json.dumps(content, ensure_ascii=False, default=str)
            record["output"] = text[:30000]
            record["output_truncated"] = len(text) > 30000

    def on_tool_error(self, error, *, run_id, **kwargs):
        record = self.tools.get(str(run_id))
        if record is not None:
            record["error"] = True

    def usage(self):
        return {
            "model_calls": len(self.model_starts), "calls_with_usage": self.known_usage_calls,
            "calls_without_usage": len(self.model_starts) - self.known_usage_calls,
            "known_input_tokens": self.input_tokens, "known_output_tokens": self.output_tokens,
            "scope": "仅含回调采集到的聊天模型用量，不含原生向量SDK和未上报用量的调用",
        }


def evaluation_metadata(dataset_sha256):
    digest = hashlib.sha256()
    for relative in ["app/agents/orchestrator.py", "app/agents/supervisor.py", "app/agents/insurance_advisor.py",
                     "app/agents/context.py", "app/agents/schemas.py", "app/agents/tools.py", "app/infra/llm.py",
                     "app/infra/reranker.py", "app/infra/embeddings.py"]:
        digest.update(relative.encode("utf-8"))
        digest.update((APP_ROOT / relative).read_bytes())
    return {
        "dataset_sha256": dataset_sha256, "workflow_sha256": digest.hexdigest(),
        "chat_model": settings.llm.chat_model, "embedding_model": settings.rag.embedding_model,
        "milvus_collection": settings.rag.milvus_collection, "embedding_dimensions": settings.rag.embedding_dimensions,
        "max_delegations": settings.supervisor.max_delegations,
        "evaluator_sha256": hashlib.sha256((APP_ROOT / "app/evaluation/evaluators.py").read_bytes()).hexdigest(),
        "evaluation_type": "rule_baseline", "target": "ChatService.chat_stream",
    }


@asynccontextmanager
async def evaluation_agent():
    # 初始化不进入实验追踪，避免读取旧会话时将真实用户的历史上传到测评记录。
    with tracing_context(enabled=False):
        from app.agents.orchestrator import init_insurance_agent
        from app.infra.checkpointer import init_checkpointer, close_checkpointer
        from app.infra.database import check_database, close_database, engine
        from app.modules.chat_thread.history import init_chat_history
        from app.rag import close_rag
        original_echo = engine.echo
        engine.echo = False
        try:
            await check_database()
            saver = await init_checkpointer()
            agent = await init_insurance_agent(saver)
            await init_chat_history(agent)
        except BaseException:
            close_rag()
            await close_checkpointer()
            await close_database()
            engine.echo = original_echo
            raise
    try:
        yield agent
    finally:
        with tracing_context(enabled=False):
            close_rag()
            await close_checkpointer()
            await close_database()
            engine.echo = original_echo


class EvaluationTarget:

    def __init__(self, agent, timeout_seconds=180):
        self.agent = agent
        self.timeout_seconds = timeout_seconds

    async def run(self, inputs):
        from app.infra.database import AsyncSessionFactory
        from app.modules.chat.service import ChatService
        from app.modules.chat.schema import ChatRequest
        from app.modules.chat_thread.models import ChatThread
        from app.modules.chat_thread.service import ChatThreadService

        thread_id = uuid4()
        collector = EvaluationTraceCollector()
        agent = self.agent.with_config({"callbacks": [collector], "tags": ["evaluation"]})
        output = {"turns": [], "error": None, "cleanup_ok": False}
        started = perf_counter()
        current_run = get_current_run_tree()
        if current_run:
            current_run.add_metadata({"thread_id": str(thread_id), "synthetic_evaluation": True})
        try:
            async with asyncio.timeout(self.timeout_seconds):
                with tracing_context(enabled=False):
                    async with AsyncSessionFactory.begin() as session:
                        session.add(ChatThread(id=thread_id, user_id=EVALUATION_USER_ID, title="LangSmith临时测评"))
                for message in inputs["messages"]:
                    turn_started = perf_counter()
                    previous_tool_ids = set(collector.tools)
                    first_token_ms = None
                    chunks = []
                    done = False
                    async with AsyncSessionFactory() as session:
                        async for event in ChatService(session, agent).chat_stream(
                            EVALUATION_USER_ID, ChatRequest(thread_id=thread_id, message=message)
                        ):
                            if event.event == "message" and event.data:
                                if first_token_ms is None:
                                    first_token_ms = round((perf_counter() - turn_started) * 1000)
                                chunks.append(event.data)
                            elif event.event == "done":
                                done = True
                    # 状态读取只用于评分，模型运行过程已由实验追踪记录。
                    with tracing_context(enabled=False):
                        snapshot = await agent.aget_state({"configurable": {"thread_id": str(thread_id)}})
                    state = snapshot.values
                    output["turns"].append({
                        "answer": "".join(chunks), "done": done,
                        "workflow": state.get("active_workflow"),
                        "delegation_count": state.get("delegation_count", 0),
                        "workers": state.get("worker_results", []),
                        "first_token_ms": first_token_ms,
                        "total_ms": round((perf_counter() - turn_started) * 1000),
                        # 按轮保存真实工具证据，不能用后续轮次的工具结果证明此前已查询。
                        "tools": [dict(tool) for tool_id, tool in collector.tools.items()
                                  if tool_id not in previous_tool_ids],
                    })
        except Exception as exc:
            # 报告不保存第三方异常正文，避免响应中意外夹带凭据或无关数据。
            output["error"] = type(exc).__name__
        finally:
            try:
                with tracing_context(enabled=False):
                    async with AsyncSessionFactory() as session:
                        if await session.get(ChatThread, thread_id):
                            await session.rollback()
                            await ChatThreadService(session, self.agent).remove(EVALUATION_USER_ID, str(thread_id))
                        else:
                            await self.agent.checkpointer.adelete_thread(str(thread_id))
                output["cleanup_ok"] = True
            except Exception as exc:
                output["cleanup_error"] = type(exc).__name__
        output["tools"] = list(collector.tools.values())
        output["usage"] = collector.usage()
        output["total_ms"] = round((perf_counter() - started) * 1000)
        return output
