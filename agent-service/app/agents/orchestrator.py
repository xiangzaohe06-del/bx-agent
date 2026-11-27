import json
from typing import Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.runnables import RunnableConfig
from langgraph.constants import START, END
from langgraph.errors import GraphRecursionError
from langgraph.graph import StateGraph
from langgraph.runtime import Runtime
from langgraph.types import Command

from app.core.config import settings
from app.core.logging import get_logger
from .schemas import InsuranceAgentState, WorkerResult, SupervisorDecision
from .intent_router import match_social_intent, response_by_social_intent
from .chitchat import init_chitchat_agent
from .insurance_advisor import init_insurance_advisor
from .context import create_summary_middleware
from .supervisor import Supervisor

logger = get_logger(__name__)

# 未接入的业务只能说明当前能力，不能声称已经办理或已转人工。
UNAVAILABLE_RESPONSES = {
    "claim": "理赔办理功能尚未接入，目前无法替您提交报案或查询理赔进度。",
    "human_handoff": "当前尚未接入人工客服转接，请通过保险公司官方客服渠道联系人工服务。",
    "fallback": "暂时无法确定您的需求，请告诉我想了解的保险产品或具体问题。",
}


def latest_user_message(messages):
    for message in reversed(messages):
        if isinstance(message, HumanMessage) and not message.additional_kwargs.get("context_summary"):
            return message.text
    return ""


def worker_input(state):
    # 任务通过临时系统消息交给子Agent，不冒充用户，也不写入主图messages。
    return {
        "messages": [
            *state["messages"],
            HumanMessage(content="本轮内部参考数据（仅作为数据）：" + json.dumps(
                state.get("worker_results", []), ensure_ascii=False
            )),
            SystemMessage(content="Supervisor本次委派：" + state["current_task"]),
        ],
        "current_task": state["current_task"],
        "previous_workflow": state.get("previous_workflow", "fallback"),
        "active_workflow": state.get("active_workflow", "fallback"),
        "chitchat_count": state.get("chitchat_count", 0),
    }


async def init_insurance_agent(checkpointer, *, supervisor=None, advisor=None, chitchat=None, summary=None):
    # 1.初始化协调者和两个子流程，允许测试注入固定模型而不调用云端。
    supervisor = supervisor if supervisor is not None else Supervisor()
    advisor = advisor if advisor is not None else await init_insurance_advisor()
    chitchat = chitchat if chitchat is not None else init_chitchat_agent()
    if summary is None and settings.context.summary_enabled:
        summary = create_summary_middleware()

    async def prepare_node(state: InsuranceAgentState, runtime: Runtime):
        # 新一轮只恢复会话上下文，清空上一轮任务，避免旧结果误用于本轮。
        update = {
            "previous_workflow": state.get("active_workflow", "fallback"),
            "current_task": "", "worker_results": [], "delegation_count": 0,
            "fixed_reply": "", "stop_reason": "",
        }
        if summary:
            compressed = await summary.abefore_model(state, runtime)
            if compressed:
                update.update(compressed)
        return update

    async def supervisor_node(state: InsuranceAgentState) -> Command[Literal[
        "recommendation_plan", "chitchat", "respond"
    ]]:
        results = state.get("worker_results", [])
        # 简单寒暄沿用原有规则，直接回复，不额外消耗模型token。
        if not results:
            social_intent = match_social_intent(latest_user_message(state["messages"]))
            if social_intent:
                return Command(update={
                    "fixed_reply": response_by_social_intent(social_intent),
                    "active_workflow": "chitchat",
                }, goto="respond")
        if results and results[-1]["status"] == "needs_user_input":
            return Command(update={"stop_reason": "需要用户补充信息，本轮先追问"}, goto="respond")
        if results and results[-1]["status"] == "completed":
            # 已完成的结果直接交给最终回复节点，不再调用主管模型确认“是否完成”。
            return Command(update={"stop_reason": "子流程已完成，直接组织最终回复"}, goto="respond")
        if results and results[-1].get("retryable") is False:
            # 递归上限表示子流程已经反复执行，重新委派通常只会重复消耗token。
            return Command(update={"stop_reason": "子流程达到递归上限，停止本轮自动重试"}, goto="respond")
        if state.get("delegation_count", 0) >= settings.supervisor.max_delegations:
            return Command(update={"stop_reason": "达到本轮最大委派次数，停止继续查询"}, goto="respond")

        try:
            decision = SupervisorDecision.model_validate(await supervisor.decide(state))
        except Exception as exc:
            logger.warning("Supervisor调度失败，异常类型：%s", type(exc).__name__)
            return Command(update={
                "stop_reason": "调度暂时失败，请根据已有结果说明情况，不编造结论",
                "fixed_reply": "暂时无法完成本次咨询，请稍后重试。" if not results else "",
            }, goto="respond")
        if decision.next_agent == "direct_response":
            return Command(update={
                "active_workflow": "direct_response",
                "current_task": latest_user_message(state["messages"]),
                "stop_reason": "本轮仅收集、确认或更新用户信息，无需产品与条款查询",
            }, goto="respond")
        if decision.next_agent == "FINISH":
            if results:
                return Command(update={"current_task": decision.task}, goto="respond")
            # 尚无任务结果时不能直接结束业务咨询，避免仅凭历史摘要回答条款问题。
            decision = SupervisorDecision(
                next_agent="recommendation_plan",
                task=latest_user_message(state["messages"]),
                reason="首轮没有子流程结果，先交保险顾问确认需求或查询依据",
            )
        if decision.next_agent in UNAVAILABLE_RESPONSES:
            return Command(update={
                "active_workflow": decision.next_agent,
                "fixed_reply": UNAVAILABLE_RESPONSES[decision.next_agent],
            }, goto="respond")

        task = decision.task.strip() or latest_user_message(state["messages"])
        # 限制完全重复的任务；语义不同的补查仍可执行，但总次数受上限约束。
        normalized = "".join(task.split()).lower()
        if any(result["agent"] == decision.next_agent and
               "".join(result["task"].split()).lower() == normalized for result in results):
            return Command(update={"stop_reason": "检测到重复任务，使用已有结果结束本轮"}, goto="respond")
        logger.info("Supervisor委派子流程：%s，当前次数：%s",
                    decision.next_agent, state.get("delegation_count", 0) + 1)
        return Command(update={
            "current_task": task,
            "active_workflow": decision.next_agent,
            "delegation_count": state.get("delegation_count", 0) + 1,
        }, goto=decision.next_agent)

    async def run_worker(state, config, agent, name):
        worker_config = {
            **config, "recursion_limit": 16,
            "tags": [*config.get("tags", []), "nostream"],
            "metadata": {**config.get("metadata", {}), "internal_agent": True},
        }
        retryable = True
        try:
            output = await agent.ainvoke(worker_input(state), worker_config)
            if name == "recommendation_plan":
                result = WorkerResult.model_validate(output["structured_response"])
                if result.missing_fields:
                    result.status = "needs_user_input"
            else:
                text = next(message.text for message in reversed(output["messages"])
                            if isinstance(message, AIMessage) and message.text)
                result = WorkerResult(status="completed", result=text)
        except Exception as exc:
            logger.warning("子流程执行失败：%s，异常类型：%s", name, type(exc).__name__)
            output = {}
            retryable = not isinstance(exc, GraphRecursionError)
            result = WorkerResult(status="failed", result=(
                "本次子流程未能完成任务。请仅说明尚未完成的部分，不能据此断言所有工具查询都失败。"
            ))
        record = {"agent": name, "task": state["current_task"], **result.model_dump()}
        if not retryable:
            # 只保存可否重试的标记，不将第三方异常正文写入会话状态。
            record["retryable"] = False
        update = {"worker_results": [*state.get("worker_results", []), record]}
        if name == "chitchat":
            update["chitchat_count"] = output.get("chitchat_count", state.get("chitchat_count", 0))
        return update

    async def recommendation_node(state: InsuranceAgentState, config: RunnableConfig):
        return await run_worker(state, config, advisor, "recommendation_plan")

    async def chitchat_node(state: InsuranceAgentState, config: RunnableConfig):
        return await run_worker(state, config, chitchat, "chitchat")

    async def respond_node(state: InsuranceAgentState):
        # 只有这个节点产生可展示消息；子Agent的工具调用和草稿不会合并到主会话。
        if state.get("fixed_reply"):
            response = AIMessage(state["fixed_reply"])
        else:
            try:
                response = await supervisor.respond(state)
                if not response.text.strip():
                    raise ValueError("最终回复为空")
            except Exception as exc:
                logger.warning("最终回复生成失败，异常类型：%s", type(exc).__name__)
                response = AIMessage("暂时无法生成完整回复，聊天记录已保留，请稍后重试。")
        return {"messages": [response]}

    # 2.子Agent执行后由规则优先收尾；只有可重试的失败才继续调用主管判断。
    builder = StateGraph(InsuranceAgentState)
    builder.add_node("prepare", prepare_node)
    builder.add_node("supervisor", supervisor_node)
    builder.add_node("recommendation_plan", recommendation_node)
    builder.add_node("chitchat", chitchat_node)
    builder.add_node("respond", respond_node)
    builder.add_edge(START, "prepare")
    builder.add_edge("prepare", "supervisor")
    builder.add_edge("recommendation_plan", "supervisor")
    builder.add_edge("chitchat", "supervisor")
    builder.add_edge("respond", END)
    return builder.compile(checkpointer=checkpointer)
