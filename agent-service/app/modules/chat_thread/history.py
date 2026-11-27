from uuid import UUID, uuid5, NAMESPACE_URL

from langchain_core.messages import AIMessage, HumanMessage
from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.infra.database import AsyncSessionFactory, engine
from app.core.logging import get_logger
from .models import ChatMessage, ChatThread, ChatProductList

logger = get_logger(__name__)


def message_records(thread_id, messages):
    """提取可展示消息；内部摘要和工具结果不进入聊天记录。"""
    records = []
    for index, message in enumerate(messages):
        if message.additional_kwargs.get("context_summary"):
            continue
        if isinstance(message, HumanMessage):
            role = "user"
        elif isinstance(message, AIMessage):
            role = "assistant"
        else:
            continue
        if not message.text:
            continue

        # 旧检查点一般已有消息ID；缺失时生成稳定ID，避免重复启动产生重复记录。
        message_id = message.id or str(uuid5(
            NAMESPACE_URL, f"{thread_id}:{index}:{role}:{message.text}"
        ))
        records.append({
            "thread_id": UUID(str(thread_id)),
            "message_id": message_id,
            "role": role,
            "content": message.text,
        })
    return records


async def archive_messages(thread_id, messages):
    """独立事务归档，成功落库后才允许压缩原消息。"""
    records = message_records(thread_id, messages)
    if not records:
        return
    async with AsyncSessionFactory.begin() as session:
        # 分批写入，避免已有长会话一次产生过多SQL参数。
        for start in range(0, len(records), 200):
            statement = insert(ChatMessage).values(records[start:start + 200])
            await session.execute(statement.on_conflict_do_nothing(
                constraint="uq_chat_messages_thread_message"
            ))


async def init_chat_history(agent):
    """启动时创建新增表，并在启用摘要前归档已有会话。"""
    async with engine.begin() as connection:
        # 只创建新增消息表，不修改已有业务表和知识库。
        await connection.run_sync(ChatMessage.__table__.create, checkfirst=True)
        await connection.run_sync(ChatProductList.__table__.create, checkfirst=True)
    async with AsyncSessionFactory() as session:
        thread_ids = (await session.scalars(select(ChatThread.id))).all()
    for thread_id in thread_ids:
        snapshot = await agent.aget_state({"configurable": {"thread_id": str(thread_id)}})
        await archive_messages(thread_id, snapshot.values.get("messages", []))
    logger.info("聊天记录表初始化完成，已检查会话数：%s", len(thread_ids))
