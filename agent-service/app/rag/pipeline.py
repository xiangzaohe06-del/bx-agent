import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from uuid import uuid4

from langchain_core.documents import Document
from langchain_milvus import Milvus
from langchain_text_splitters import (
    MarkdownHeaderTextSplitter,
    RecursiveCharacterTextSplitter,
)
from mineru import MinerU

from app.core.config import APP_ROOT, settings
from app.infra.database import AsyncSessionFactory
from app.infra.llm import create_chat_model
from app.modules.product.models import Product
from app.rag.models import ParentChunk
from app.rag.repository import ParentChunkRepository
from app.core.logging import get_logger
logger = get_logger(__name__)


MARKDOWN_OPTIMIZATION_PROMPT = """
下面这份 Markdown 文档是从保险条款 PDF 解析得来，由于 PDF 中的各个小节是以表格形式存在，所以解析时出现错乱。你分析内容，帮我转为格式正确的 Markdown，特别是标题编号要正确。
- 标题等级要从 1 级标题开始，逐层增加，目录和文档名不计入标题等级。
- 输出结果中不要包含正文开始之前的部分。
- 输出结果只包含 Markdown 正文，不要解释或代码围栏。
- 不要修改保险条款。
- 必须保留全部正文，不得总结、省略章节或表格内容。
- 输入可能是文档中间的一段，保留原章节编号，不要从头重新编号，不要补写前后文或重复目录。
""".strip()


class RAGPipeline:

    OPTIMIZATION_CHARS = 6000
    MIN_RETRY_CHARS = 1000
    EMBEDDING_BATCH_SIZE = 16
    CACHE_VERSION = "segmented-v1"

    def __init__(self, vector_store: Milvus) -> None:
        self._vector_store = vector_store

        # 1.初始化文档解析工具
        self._mineru = MinerU(settings.rag.mineru_token)

        self._model = create_chat_model(max_tokens=10000)

        # 2.初始化文档切分工具
        self._markdown_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
                ("####", "h4"),
            ],
            strip_headers=True,
        )
        self._child_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50,
            separators=["\n\n", "\n", "。", "；", ";", "，", ","],
            keep_separator="end",
        )
        logger.info("RAG Pipeline 初始化完成✅️")

    async def ingest(self, product: Product) -> None:
        # 1.解析并优化保险条款
        markdown = self._parse_and_optimize(product.clause_name)

        # 2.切分父块和子块
        parent_chunks, child_chunks = self._chunk(markdown, product)
        if not parent_chunks or not child_chunks:
            raise RuntimeError(f"条款《{product.clause_name}》未生成有效知识块，已停止入库。")

        # 3.子块存入Milvus
        self._save_child_chunks(product.id, child_chunks)

        # 4.父块存入PostgreSQL
        await self._save_parent_chunks(product.id, parent_chunks)

    async def is_ingested(self, product_id: int) -> bool:
        """父块及其关联向量都存在时，才认为该产品已完成入库。"""
        from sqlalchemy import select

        async with AsyncSessionFactory() as session:
            parent_ids = set(str(value) for value in await session.scalars(
                select(ParentChunk.id).where(ParentChunk.product_id == product_id)
            ))
        if not parent_ids or not self._vector_store.client.has_collection(
            self._vector_store.collection_name
        ):
            return False
        children = self._vector_store.client.query(
            collection_name=self._vector_store.collection_name,
            filter=f"product_id == {product_id}",
            output_fields=["parent_id"],
            limit=16384,
            consistency_level="Strong",
        )
        child_parent_ids = {str(child.get("parent_id")) for child in children}
        return parent_ids == child_parent_ids

    @staticmethod
    def _split_markdown(markdown: str, limit: int) -> list[str]:
        """优先在段落或行边界切分，原文不丢字，也不重复重叠内容。"""
        parts = []
        while len(markdown) > limit:
            boundary = markdown.rfind("\n\n", limit // 2, limit)
            if boundary >= 0:
                boundary += 2
            else:
                boundary = markdown.rfind("\n", limit // 2, limit)
                boundary = boundary + 1 if boundary >= 0 else limit
            parts.append(markdown[:boundary])
            markdown = markdown[boundary:]
        if markdown:
            parts.append(markdown)
        return parts

    @staticmethod
    def _write_cache(path: Path, text: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f"{path.name}.{uuid4().hex}.tmp")
        try:
            temporary.write_text(text, encoding="utf-8")
            temporary.replace(path)
        finally:
            temporary.unlink(missing_ok=True)

    def _parse_and_optimize(self, clause_name: str) -> str:
        # 1.使用MinerU解析PDF
        file_path = Path(APP_ROOT / "data" / "raw" / "kb" / clause_name)
        file_hash = hashlib.sha256(file_path.read_bytes()).hexdigest()
        cache_dir = APP_ROOT / "data" / "cache" / "rag" / file_hash
        parsed_path = cache_dir / "parsed.md"
        if parsed_path.exists():
            source = parsed_path.read_text(encoding="utf-8")
            logger.info("复用PDF解析缓存：%s", clause_name)
        else:
            logger.info("正在解析PDF：%s", clause_name)
            source = self._mineru.extract(str(file_path)).markdown
        if not isinstance(source, str) or not source.strip():
            raise RuntimeError(f"条款《{clause_name}》解析结果为空，已停止入库。")
        if not parsed_path.exists():
            self._write_cache(parsed_path, source)

        # 缓存随模型、提示词和切分版本变化，避免复用旧规则的整理结果。
        fingerprint = hashlib.sha256(
            (settings.llm.chat_model + MARKDOWN_OPTIMIZATION_PROMPT + self.CACHE_VERSION).encode()
        ).hexdigest()
        parts_dir = cache_dir / fingerprint
        parts = self._split_markdown(source, self.OPTIMIZATION_CHARS)
        logger.info("条款分为 %s 段，每段最多 %s 字符：%s", len(parts), self.OPTIMIZATION_CHARS, clause_name)

        def optimize(item):
            index, text = item
            logger.info("整理进度 %s/%s：%s", index + 1, len(parts), clause_name)
            return self._optimize_part(text, clause_name, parts_dir)

        # 最多三个请求并行；map 按输入顺序合并，失败时已完成段落仍保存在缓存中。
        with ThreadPoolExecutor(max_workers=3) as executor:
            optimized = list(executor.map(optimize, enumerate(parts)))
        return "\n\n".join(optimized)

    def _optimize_part(self, source: str, clause_name: str, parts_dir: Path) -> str:
        part_hash = hashlib.sha256(source.encode()).hexdigest()
        cache_path = parts_dir / f"{part_hash}.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                if cached.get("finish_reason") == "stop" and isinstance(cached.get("text"), str) and cached["text"].strip():
                    return cached["text"]
            except (ValueError, OSError, AttributeError):
                logger.warning("段落缓存损坏，将重新整理：%s", clause_name)

        response = self._model.invoke(
            [
                {"role": "system", "content": MARKDOWN_OPTIMIZATION_PROMPT},
                {"role": "user", "content": source},
            ]
        )
        finish_reason = response.response_metadata.get("finish_reason")
        if finish_reason == "length":
            if len(source) > self.MIN_RETRY_CHARS:
                logger.warning("段落输出截断，自动拆小重试：%s（%s字符）", clause_name, len(source))
                smaller = self._split_markdown(source, max(1, len(source) // 2))
                markdown = "\n\n".join(self._optimize_part(part, clause_name, parts_dir) for part in smaller)
                self._write_cache(cache_path, json.dumps({"finish_reason": "stop", "text": markdown}, ensure_ascii=False))
                return markdown
            raise RuntimeError(
                f"条款《{clause_name}》整理达到模型输出长度上限，内容可能被截断，"
                "已停止入库。请将文档分段后重试。"
            )
        if finish_reason != "stop":
            raise RuntimeError(
                f"条款《{clause_name}》整理未正常结束（finish_reason={finish_reason!r}），"
                "已停止入库。"
            )
        markdown = response.text.strip()
        if not markdown:
            raise RuntimeError(f"条款《{clause_name}》整理结果为空，已停止入库。")
        self._write_cache(cache_path, json.dumps({"finish_reason": "stop", "text": markdown}, ensure_ascii=False))
        return markdown

    def _chunk(
        self,
        markdown: str,
        product: Product,
    ) -> tuple[list[ParentChunk], list[Document]]:
        # 1.按照Markdown标题切分父块
        sections = self._markdown_splitter.split_text(markdown)
        parent_chunks: list[ParentChunk] = []
        child_chunks: list[Document] = []

        # 2.遍历父块，继续切分子块
        for section in sections:
            section_path = list(section.metadata.values())
            parent = ParentChunk(
                id=uuid4(),
                product_id=product.id,
                clause_name=product.clause_name,
                section_path=section_path,
                content=section.page_content,
            )
            parent_chunks.append(parent)

            # 子块通过parent_id关联父块
            section.metadata = {
                "product_id": product.id,
                "parent_id": str(parent.id),
            }
            children = self._child_splitter.split_documents([section])

            # 把章节路径拼接到子块正文中，补充语义上下文
            section_header = "\n".join(section_path) + "\n"
            for child in children:
                child.page_content = section_header + child.page_content
            child_chunks.extend(children)

        return parent_chunks, child_chunks

    def _save_child_chunks(
        self,
        product_id: int,
        child_chunks: list[Document],
    ) -> None:
        # Collection存在时，先删除该产品的旧子块
        if self._vector_store.client.has_collection(
            self._vector_store.collection_name
        ):
            self._vector_store.delete(expr=f"product_id == {product_id}")
        # 控制单次文本数量，适用于本机或云端向量模型。
        for start in range(0, len(child_chunks), self.EMBEDDING_BATCH_SIZE):
            batch = child_chunks[start:start + self.EMBEDDING_BATCH_SIZE]
            self._vector_store.add_documents(batch)
            logger.info("产品%s向量入库进度：%s/%s", product_id, min(start + len(batch), len(child_chunks)), len(child_chunks))

    async def _save_parent_chunks(
        self,
        product_id: int,
        parent_chunks: list[ParentChunk],
    ) -> None:
        async with AsyncSessionFactory() as session:
            async with session.begin():
                repository = ParentChunkRepository(session)
                # 删除旧父块
                await repository.delete_by_product(product_id)
                # 添加新父块
                repository.add_all(parent_chunks)

    def close(self):
        """关闭mineru连接"""

        self._mineru.close()
