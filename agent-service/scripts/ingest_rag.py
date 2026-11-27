"""RAG知识库构建脚本：查询所有保险产品，解析条款并写入向量库和数据库"""

import asyncio

from app.core.logging import configure_logging, get_logger
from app.infra.database import AsyncSessionFactory, close_database
from app.modules.product.service import ProductService
from app.rag import rag_pipeline, close_rag

logger = get_logger(__name__)


async def main() -> None:
    completed, skipped, failed = [], [], []
    try:
        # 4.查询所有products
        async with AsyncSessionFactory() as session:
            service = ProductService(session)
            products = await service.list_product()
        logger.info(f"查询到 {len(products)} 个产品")

        # 5.逐个产品构建知识库
        for product in products:
            try:
                if await rag_pipeline.is_ingested(product.id):
                    skipped.append(product.name)
                    logger.info("跳过已入库产品：%s", product.name)
                    continue
                logger.info(f"正在构建知识库: {product.name}...")
                await rag_pipeline.ingest(product)
                if not await rag_pipeline.is_ingested(product.id):
                    raise RuntimeError("入库后父块与向量关联校验未通过")
                completed.append(product.name)
                logger.info(f"完成: {product.name}")
            except Exception as exc:
                failed.append(product.name)
                logger.error("产品导入失败，继续处理下一款：%s，原因：%s", product.name, exc)

        logger.info("导入结束：完成%s款，跳过%s款，失败%s款", len(completed), len(skipped), len(failed))
        if failed:
            logger.error("失败产品：%s；重新运行同一命令可复用缓存继续导入。", "、".join(failed))
            raise SystemExit(1)
        logger.info("RAG知识库构建完成✅")
    finally:
        close_rag()
        await close_database()


if __name__ == "__main__":
    configure_logging("INFO")
    asyncio.run(main())
