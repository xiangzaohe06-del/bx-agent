from langchain_core.messages import AIMessage, SystemMessage
from langgraph.constants import START, END
from langgraph.graph import StateGraph

from app.infra.llm import create_chat_model
from app.core.logging import get_logger
from .schemas import InsuranceAgentState

logger = get_logger(__name__)

SYSTEM_MESSAGE = """
你是《安心保》商城的智能客服，负责处理与保险业务无关的普通聊天。

要求：
1. 回复友好、自然，最多两句话。
2. 不假装有个人经历、情感、实时信息或外部能力。
3. 不回答医疗、法律、投资、政治等高风险专业建议。
4. 每次回复都自然引导用户回到保险产品、投保或理赔业务。
5. 超出能力范围时，如实说明无法提供。
""".strip()


# 1.定义State
class ChitchatState(InsuranceAgentState):
    pass     # 闲聊计数沿用主会话State，由Supervisor回传更新

# 2.定义Workflow

MAX_COUNT = 2   # 最大闲聊轮次

class ChitChatWorkflow:

    def __init__(self):
        self.model = create_chat_model().with_config({
            "tags": ["nostream"], "metadata": {"internal_agent": True}
        })


    async def chitchat(self, state: ChitchatState):

        # 取出State中的闲聊次数
        count = state.get("chitchat_count", 0)

        # 取出上一轮次的工作流，如果上一轮次不是闲聊，则重置count
        if state.get("previous_workflow") != 'chitchat':
            count = 0

        # 判断count是否超出了闲聊的最大轮次
        if count >= MAX_COUNT:
            logger.info("闲聊次数超过最大值，直接返回固定结果。")
            return {"messages": [AIMessage("我主要协助处理保险产品、投保和理赔咨询，请告诉我想办理或了解的事项。")]}

        logger.info(f"当前闲聊次：{count}")
        # 没超过闲聊最大轮次，可以正常对话
        task = state.get("current_task", "处理本轮普通聊天")
        response = await self.model.ainvoke([
            SystemMessage(SYSTEM_MESSAGE),
            *state['messages'],
            SystemMessage(content=f"本次委派任务：{task}")
        ])

        # 返回结果，更新Message、更新count
        return {
            "messages": [response],
            "chitchat_count": count + 1
        }

    def build_chitchat_graph(self):

        # 创建Builder
        builder = StateGraph(ChitchatState)

        # 添加节点
        builder.add_node("chitchat", self.chitchat)

        # 添加边
        builder.set_entry_point("chitchat")     # START -> chitchat
        builder.set_finish_point("chitchat")    # chitchat -> END

        # 编译
        # 子流程使用Supervisor传入的上下文和计数，不独立累积消息历史。
        return builder.compile(checkpointer=False)

def init_chitchat_agent():
    return ChitChatWorkflow().build_chitchat_graph()
