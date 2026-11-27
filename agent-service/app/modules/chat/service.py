from uuid import uuid4

from fastapi.sse import ServerSentEvent
from langchain.messages import HumanMessage
from langchain_core.messages import AIMessage
from langgraph.graph.state import CompiledStateGraph
from sqlalchemy.ext.asyncio import AsyncSession

from app.modules.chat_thread.repository import ChatThreadRepository
from app.modules.chat_thread.exceptions import ChatThreadNotFound
from .schema import ChatRequest
from app.agents.context import is_internal_message
from app.modules.chat_thread.history import archive_messages


class ChatService:

    def __init__(self, session: AsyncSession, agent: CompiledStateGraph):
        self.agent = agent
        self.repository = ChatThreadRepository(session)

    async def chat_stream(self, user_id: int, request: ChatRequest):
        """实现Agent流式对话的函数"""

        # 1.校验，判断当前thread_id是否属于当前用户
        chat_thread = await self.repository.required_owner(user_id, request.thread_id)
        if not chat_thread:
            # 指定会话不属于当前用户
            raise ChatThreadNotFound()

        # 2.组织消息
        _input = {"messages": [HumanMessage(request.message, id=str(uuid4()))]}

        # 3.组织config，实现会话记忆
        config = {"configurable": {"thread_id": request.thread_id}}

        # 先归档历史和本轮用户消息，避免后续摘要改变页面可查看的原文。
        snapshot = await self.agent.aget_state(config)
        await archive_messages(request.thread_id, snapshot.values.get("messages", []))
        await archive_messages(request.thread_id, _input["messages"])

        # 4.调用Agent
        stream = await self.agent.astream_events(
            _input,
            config,
            version="v3"
        )

        # 5.返回结果
        async for event in stream:
            method = event['method']
            # 判断事件类型，可以是messages、interrupts等等
            if method == 'messages':
                data = event['params']['data'][0]
                metadata = event['params']['data'][1]
                # 内部摘要不属于客服回复，即使收到模型事件也不转发给前端。
                if is_internal_message(data, metadata):
                    continue
                # Supervisor统一对外回复，只转发respond节点，避免内部草稿混入SSE。
                if metadata.get("langgraph_node") != "respond":
                    continue
                if isinstance(data, AIMessage):
                    yield ServerSentEvent(data=data.text, event="message")
                # 状态替换还会产生RemoveMessage等对象，只处理文本增量字典。
                elif isinstance(data, dict) and data.get('delta') and data['delta'].get('text'):
                    yield ServerSentEvent(data=data['delta'].get('text'), event="message")

        # 固定回复和闲聊不经过保险顾问中间件，结束时同样需要保存。
        snapshot = await self.agent.aget_state(config)
        await archive_messages(request.thread_id, snapshot.values.get("messages", []))
        yield ServerSentEvent(data="END", event="done")
