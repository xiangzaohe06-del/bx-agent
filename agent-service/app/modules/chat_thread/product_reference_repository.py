from uuid import UUID

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert

from app.infra.database import AsyncSessionFactory
from app.modules.product.models import Product
from .models import ChatMessage, ChatProductList


def product_list_dict(record):
    return {
        "list_id": record.id, "message_id": record.message_id, "turn_id": record.turn_id,
        "group_index": record.group_index, "label": record.label, "products": record.products,
    }


class ProductReferenceRepository:
    """所有产品组查询都限定thread_id，避免跨会话混用。"""

    async def list_products(self):
        async with AsyncSessionFactory() as session:
            rows = (await session.execute(select(
                Product.id, Product.name, Product.category, Product.min_premium
            ).order_by(Product.id))).all()
        return [{
            "product_id": row.id, "name": row.name, "category": row.category,
            "min_premium": str(row.min_premium) if row.min_premium is not None else None,
        } for row in rows]

    async def list_history(self, thread_id):
        # 只在旧会话首次接入或归档失败后回填，不在每轮向模型发送全部历史。
        async with AsyncSessionFactory() as session:
            return (await session.scalars(select(ChatMessage).where(
                ChatMessage.thread_id == UUID(str(thread_id))
            ).order_by(ChatMessage.id))).all()

    async def save(self, thread_id, message_id, turn_id, groups):
        if not groups:
            return []
        async with AsyncSessionFactory.begin() as session:
            statement = insert(ChatProductList).values([{
                "thread_id": UUID(str(thread_id)), "message_id": message_id, "turn_id": turn_id,
                "group_index": index, "label": group["label"][:200], "products": group["products"],
            } for index, group in enumerate(groups)])
            await session.execute(statement.on_conflict_do_nothing(
                constraint="uq_chat_product_lists_message_group"
            ))
        return await self.for_message(thread_id, message_id)

    async def _select(self, thread_id, *conditions, first=False, limit=None):
        statement = select(ChatProductList).where(
            ChatProductList.thread_id == UUID(str(thread_id)), *conditions
        ).order_by(ChatProductList.id.asc() if first else ChatProductList.id.desc())
        if limit:
            statement = statement.limit(limit)
        async with AsyncSessionFactory() as session:
            rows = (await session.scalars(statement)).all()
        return [product_list_dict(row) for row in rows]

    async def recent(self, thread_id, limit=3):
        return list(reversed(await self._select(thread_id, limit=limit)))

    async def get(self, thread_id, list_id):
        rows = await self._select(thread_id, ChatProductList.id == list_id, limit=1)
        return rows[0] if rows else None

    async def first(self, thread_id):
        rows = await self._select(thread_id, first=True, limit=1)
        return rows[0] if rows else None

    async def before(self, thread_id, list_id):
        rows = await self._select(thread_id, ChatProductList.id < list_id, limit=1)
        return rows[0] if rows else None

    async def for_message(self, thread_id, message_id):
        return await self._select(thread_id, ChatProductList.message_id == message_id, first=True)
