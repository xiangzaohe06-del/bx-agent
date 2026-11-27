import math

from dashscope import MultiModalEmbedding
from langchain_core.embeddings import Embeddings

from app.core.config import settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class BailianVLTextEmbeddings(Embeddings):
    """通过百炼原生多模态接口，为每条文本生成独立向量。"""

    BATCH_SIZE = 20
    SUPPORTED_DIMENSIONS = {256, 512, 768, 1024, 1536, 2048, 2560}

    def __init__(self, model: str = "qwen3-vl-embedding", dimensions: int = 1024):
        if model != "qwen3-vl-embedding":
            raise ValueError("当前云端适配器仅支持 qwen3-vl-embedding")
        if dimensions not in self.SUPPORTED_DIMENSIONS:
            raise ValueError("qwen3-vl-embedding 不支持该向量维度")
        self.model = model
        self.dimensions = dimensions

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if any(not isinstance(text, str) or not text.strip() for text in texts):
            raise ValueError("向量化文本不能为空")
        vectors = []
        for start in range(0, len(texts), self.BATCH_SIZE):
            batch = texts[start:start + self.BATCH_SIZE]
            response = MultiModalEmbedding.call(
                model=self.model,
                input=[{"text": text} for text in batch],
                dimension=self.dimensions,
                enable_fusion=False,
                api_key=settings.llm.dashscope_api_key,
                base_address=settings.rag.embedding_base_url,
                request_timeout=60,
            )
            if response.status_code != 200:
                raise RuntimeError(
                    f"百炼向量化失败：HTTP {response.status_code}，错误码 {response.code}"
                )
            items = (response.output or {}).get("embeddings", [])
            indices = [item.get("index") for item in items]
            if (len(items) != len(batch)
                    or any(type(index) is not int for index in indices)
                    or set(indices) != set(range(len(batch)))):
                raise RuntimeError("百炼向量结果数量或编号异常，停止写入")
            ordered = sorted(items, key=lambda item: item["index"])
            for item in ordered:
                vector = item.get("embedding", [])
                if (len(vector) != self.dimensions
                        or any(type(value) not in (int, float) or not math.isfinite(value) for value in vector)):
                    raise RuntimeError("百炼向量维度或数值异常，停止写入")
                vectors.append(vector)
            logger.info("云端向量化完成：%s条，%s维", len(batch), self.dimensions)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]


def create_embeddings() -> Embeddings:
    if settings.rag.embedding_provider == "dashscope":
        return BailianVLTextEmbeddings(
            model=settings.rag.embedding_model,
            dimensions=settings.rag.embedding_dimensions,
        )
    from langchain_ollama import OllamaEmbeddings

    return OllamaEmbeddings(
        model=settings.rag.embedding_model,
        dimensions=settings.rag.embedding_dimensions,
    )
