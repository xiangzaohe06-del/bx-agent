from typing import Literal, NotRequired

from langgraph.graph import MessagesState
from pydantic import BaseModel, Field

# 工作流名称常量，兼容原有会话中的工作流字段。
Intent = Literal[
    "chitchat",
    "recommendation_plan",
    "claim",
    "human_handoff",
    "fallback",
    "direct_response",
]


class SupervisorDecision(BaseModel):
    """Supervisor决定下一步委派给谁，或结束本轮任务。"""

    next_agent: Literal[
        "recommendation_plan", "chitchat", "claim", "human_handoff", "fallback", "direct_response", "FINISH"
    ] = Field(description="需求收集、信息确认或更新选择direct_response；需要产品或条款依据时委派；有结果后可FINISH")
    task: str = Field(default="", description="委派的具体任务，或最终回复需要完成的事项")
    reason: str = Field(default="", description="选择该步骤的简短依据")


class WorkerResult(BaseModel):
    """子Agent的内部任务结果，不直接写入用户聊天记录。"""

    status: Literal["completed", "needs_user_input", "failed"]
    result: str = Field(min_length=1, description="任务结论，必须依据用户信息和工具结果")
    missing_fields: list[str] = Field(default_factory=list, description="需要用户补充的信息")
    sources: list[str] = Field(default_factory=list, description="实际查询到的产品ID、条款章节等来源")


class InsuranceAgentState(MessagesState):
    """保险顾问主Graph状态"""

    previous_workflow: Intent
    active_workflow: Intent
    # 保留原有工作流字段兼容旧会话；新增字段在每轮prepare节点中初始化。
    chitchat_count: NotRequired[int]
    current_task: NotRequired[str]
    worker_results: NotRequired[list[dict]]
    delegation_count: NotRequired[int]
    fixed_reply: NotRequired[str]
    stop_reason: NotRequired[str]
    # 产品组和关注对象跨轮保留，本轮指代结果单独更新，不混入公开messages。
    product_context: NotRequired[dict]
    reference_resolution: NotRequired[dict]
