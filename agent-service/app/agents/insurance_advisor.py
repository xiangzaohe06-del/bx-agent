from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langgraph.graph.state import CompiledStateGraph

from app.infra.llm import create_chat_model
from app.core.config import settings
from app.core.logging import get_logger
from .tools import query_candidate_products, retrieve_insurance_clause
from .context import create_summary_middleware
from .schemas import WorkerResult

logger = get_logger(__name__)

SYSTEM_PROMPT = """
你是“安心保”Supervisor委派的保险顾问，负责向Supervisor返回内部任务结果。
你需要使用专业、准确、容易理解的语言完成被委派的具体任务。
当信息不足时，应先向用户追问，不要编造保险产品或保障内容。
当用户询问保险条款问题时，必须调用工具查询条款，然后根据条款回答，不能自己编造。
完成后使用WorkerResult返回结果：completed表示已完成，needs_user_input表示需要用户补充资料，
failed表示查询失败。missing_fields只填写确实缺少的信息，sources只填写实际查到的来源。
信息不足时直接返回需要补充的内容，不要反复查询，也不要假装已经完成投保或人工转接。
"""


# agent在整个系统中，只初始化一次，
async def init_insurance_advisor() -> CompiledStateGraph:
    # 1.初始化模型
    # 内部消息标记由Supervisor调用子流程时统一传入，保留原始模型的工具绑定能力。
    model = create_chat_model()

    # 2.创建Agent，绑定model、tools、Checkpointer
    agent = create_agent(
        model=model,
        tools=[query_candidate_products, retrieve_insurance_clause],
        # 子Agent按本次委派运行，内部消息不进入主会话和聊天记录表。
        middleware=[create_summary_middleware(archive_history=False)] if settings.context.summary_enabled else [],
        response_format=ToolStrategy(WorkerResult, handle_errors=False),
        checkpointer=False,
        system_prompt=SYSTEM_PROMPT
    )

    logger.info("保险顾问Agent初始化完毕~✅️")
    return agent
