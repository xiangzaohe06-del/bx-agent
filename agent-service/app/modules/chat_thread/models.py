from uuid import UUID, uuid4

from sqlalchemy import BigInteger, Integer, String, Text, ForeignKey, UniqueConstraint
from sqlalchemy.dialects.postgresql import UUID as PG_UUID, JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.common.models import Base, CreateAtMixin, UpdateAtMixin


class ChatThread(Base, CreateAtMixin, UpdateAtMixin):
    """客服会话"""

    __tablename__ = "chat_threads"

    id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        primary_key=True,
        default=uuid4,
    )
    user_id: Mapped[int] = mapped_column(BigInteger)
    title: Mapped[str] = mapped_column(
        String(200),
        default="新会话",
        server_default="新会话",
    )


class ChatMessage(Base, CreateAtMixin):
    """供页面展示的原始消息，不随模型上下文压缩而删除。"""

    __tablename__ = "chat_messages"
    __table_args__ = (
        UniqueConstraint("thread_id", "message_id", name="uq_chat_messages_thread_message"),
    )

    # 自增编号用于保持归档顺序，同一条消息重复归档时通过唯一约束去重。
    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    thread_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True),
        ForeignKey("chat_threads.id", ondelete="CASCADE"),
        index=True,
    )
    message_id: Mapped[str] = mapped_column(Text)
    role: Mapped[str] = mapped_column(String(20))
    content: Mapped[str] = mapped_column(Text)


class ChatProductList(Base, CreateAtMixin):
    """按最终回复的实际展示顺序保存产品组，旧组不随上下文压缩而删除。"""

    __tablename__ = "chat_product_lists"
    __table_args__ = (
        UniqueConstraint("thread_id", "message_id", "group_index", name="uq_chat_product_lists_message_group"),
    )

    id: Mapped[int] = mapped_column(BigInteger, primary_key=True, autoincrement=True)
    thread_id: Mapped[UUID] = mapped_column(
        PG_UUID(as_uuid=True), ForeignKey("chat_threads.id", ondelete="CASCADE"), index=True,
    )
    message_id: Mapped[str] = mapped_column(Text)
    turn_id: Mapped[str] = mapped_column(Text)
    group_index: Mapped[int] = mapped_column(Integer)
    label: Mapped[str] = mapped_column(String(200), default="产品列表")
    products: Mapped[list[dict]] = mapped_column(JSONB)
