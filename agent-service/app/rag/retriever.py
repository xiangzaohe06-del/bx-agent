from langchain_milvus import Milvus

from app.infra.database import AsyncSessionFactory
from app.infra.reranker import ChatModelReranker
from .repository import ParentChunkRepository


class ParentChunkRetriever:

    MAX_CANDIDATES = 10

    def __init__(self, vector_store: Milvus):
        self._vector_store = vector_store
        self._reranker = ChatModelReranker()

    async def retrieve(self, query: str, k: int, product_id: int):
        if k <= 0:
            return []
        k = min(k, self.MAX_CANDIDATES)
        # 1.仅在 Milvus 中做混合召回，不调用其云端重排服务。
        child_chunks = await self._vector_store.asimilarity_search(
            query,
            k=self.MAX_CANDIDATES,
            expr=f"product_id == {product_id}",
        )

        # 2.由 Python 调用百炼聊天模型重排，复用项目中的 Key 和模型配置。
        child_chunks = await self._reranker.rerank(query, child_chunks, k)

        # 2.从子块中拿到parent_id，并且去重
        parent_ids = list(
            dict.fromkeys(
                child.metadata["parent_id"] for child in child_chunks
                if child.metadata.get("parent_id")
            )
        )
        if not parent_ids:
            return []

        # 3.查询父块
        async with AsyncSessionFactory() as session:
            repo = ParentChunkRepository(session)
            parents = await repo.query_by_ids(parent_ids)


        # 4.把parents按照检索回来的顺序排序

        # 4.1.把parents变成一个字典，key是parent_id，值是parent
        parent_id_dict = {
            str(parent.id): parent
            for parent in parents
        }

        # 4.2.遍历parent_ids，逐一找到parent，放到列表中
        parent_chunks = [
            parent_id_dict[id]
            for id in parent_ids
            if id in parent_id_dict
        ]

        return parent_chunks
