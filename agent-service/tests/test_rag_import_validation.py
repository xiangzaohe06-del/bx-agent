import pathlib
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langchain_core.messages import AIMessage

# 包初始化会创建客户端；离线测试用替身阻止数据库和外部服务连接。
with patch('langchain_milvus.Milvus'), patch('mineru.MinerU'), patch('app.infra.llm.create_chat_model'):
    from app.rag.pipeline import RAGPipeline


class ImportValidationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = pathlib.Path(directory.name)
        self.root_patch = patch('app.rag.pipeline.APP_ROOT', self.root)
        self.root_patch.start()
        self.addCleanup(self.root_patch.stop)
        pdf = self.root / 'data' / 'raw' / 'kb' / 'test-clause.pdf'
        pdf.parent.mkdir(parents=True)
        pdf.write_bytes(b'test-pdf-content')
        self.pipeline = object.__new__(RAGPipeline)
        self.pipeline._mineru = Mock()
        self.pipeline._mineru.extract.return_value = SimpleNamespace(markdown='# 条款\n完整原文')
        self.pipeline._model = Mock()
        self.pipeline._chunk = Mock(return_value=([object()], [object()]))
        self.pipeline._save_child_chunks = Mock()
        self.pipeline._save_parent_chunks = AsyncMock()
        self.product = SimpleNamespace(id=12, clause_name='test-clause.pdf')

    def assert_no_writes(self):
        self.pipeline._chunk.assert_not_called()
        self.pipeline._save_child_chunks.assert_not_called()
        self.pipeline._save_parent_chunks.assert_not_called()

    async def test_normal_completion_can_be_imported(self):
        self.pipeline._model.invoke.return_value = AIMessage(
            content='  # 条款\n完整正文  ', response_metadata={'finish_reason': 'stop'}
        )
        await self.pipeline.ingest(self.product)
        self.assertEqual(self.pipeline._chunk.call_args.args[0], '# 条款\n完整正文')
        self.pipeline._save_child_chunks.assert_called_once()
        self.pipeline._save_parent_chunks.assert_awaited_once()

    async def test_truncated_output_never_replaces_existing_knowledge(self):
        self.pipeline._model.invoke.return_value = AIMessage(
            content='不完整正文', response_metadata={'finish_reason': 'length'}
        )
        with self.assertRaisesRegex(RuntimeError, '截断'):
            await self.pipeline.ingest(self.product)
        self.assert_no_writes()

    async def test_abnormal_or_missing_finish_reason_prevents_writes(self):
        for reason in [None, 'content_filter', 'tool_calls']:
            with self.subTest(reason=reason):
                metadata = {} if reason is None else {'finish_reason': reason}
                self.pipeline._model.invoke.return_value = AIMessage(content='正文', response_metadata=metadata)
                with self.assertRaisesRegex(RuntimeError, '未正常结束'):
                    await self.pipeline.ingest(self.product)
                self.assert_no_writes()

    async def test_empty_extraction_does_not_call_model_or_write(self):
        self.pipeline._mineru.extract.return_value = SimpleNamespace(markdown=' \n ')
        with self.assertRaisesRegex(RuntimeError, '解析结果为空'):
            await self.pipeline.ingest(self.product)
        self.pipeline._model.invoke.assert_not_called()
        self.assert_no_writes()

    async def test_empty_model_response_prevents_writes(self):
        self.pipeline._model.invoke.return_value = AIMessage(
            content=' \n ', response_metadata={'finish_reason': 'stop'}
        )
        with self.assertRaisesRegex(RuntimeError, '整理结果为空'):
            await self.pipeline.ingest(self.product)
        self.assert_no_writes()

    async def test_split_preserves_all_characters_without_overlap(self):
        source = ('# 原章节\n一段条款正文。\n\n' * 1500) + '最后一行'
        parts = self.pipeline._split_markdown(source, 6000)
        self.assertEqual(''.join(parts), source)
        self.assertTrue(all(0 < len(part) <= 6000 for part in parts))
        self.assertGreater(len(parts), 1)

    async def test_cached_document_avoids_repeated_external_calls(self):
        self.pipeline._model.invoke.return_value = AIMessage(
            content='# 条款\n完整正文', response_metadata={'finish_reason': 'stop'}
        )
        first = self.pipeline._parse_and_optimize(self.product.clause_name)
        second = self.pipeline._parse_and_optimize(self.product.clause_name)
        self.assertEqual(first, second)
        self.pipeline._mineru.extract.assert_called_once()
        self.pipeline._model.invoke.assert_called_once()

    async def test_truncated_segment_retries_smaller_parts_in_order(self):
        source = '甲' * 3000
        self.pipeline._mineru.extract.return_value = SimpleNamespace(markdown=source)
        def respond(messages):
            text = messages[1]['content']
            if len(text) > 1500:
                return AIMessage(content='截断内容', response_metadata={'finish_reason': 'length'})
            return AIMessage(content=text, response_metadata={'finish_reason': 'stop'})
        self.pipeline._model.invoke.side_effect = respond
        output = self.pipeline._parse_and_optimize(self.product.clause_name)
        self.assertEqual(output.replace('\n', ''), source)
        # 两个相同的小段可以共享同一份段落缓存。
        self.assertEqual(self.pipeline._model.invoke.call_count, 2)

    async def test_resume_reuses_completed_segments_after_failure(self):
        self.pipeline.OPTIMIZATION_CHARS = 6
        source = '甲甲甲甲甲甲乙乙乙乙乙乙'
        self.pipeline._mineru.extract.return_value = SimpleNamespace(markdown=source)
        def first_attempt(messages):
            text = messages[1]['content']
            if text.startswith('乙'):
                raise TimeoutError('temporary outage')
            return AIMessage(content=text, response_metadata={'finish_reason': 'stop'})
        self.pipeline._model.invoke.side_effect = first_attempt
        with self.assertRaises(TimeoutError):
            self.pipeline._parse_and_optimize(self.product.clause_name)
        self.pipeline._model.invoke.reset_mock()
        self.pipeline._model.invoke.side_effect = lambda messages: AIMessage(
            content=messages[1]['content'], response_metadata={'finish_reason': 'stop'}
        )
        output = self.pipeline._parse_and_optimize(self.product.clause_name)
        self.assertEqual(output.replace('\n', ''), source)
        self.pipeline._mineru.extract.assert_called_once()
        self.pipeline._model.invoke.assert_called_once()
        self.assertTrue(self.pipeline._model.invoke.call_args.args[0][1]['content'].startswith('乙'))

    async def test_completion_requires_both_parent_and_vector_records(self):
        self.pipeline._vector_store = Mock()
        self.pipeline._vector_store.collection_name = 'test_collection'
        self.pipeline._vector_store.client.has_collection.return_value = True
        session = Mock()
        session.scalars = AsyncMock(return_value=['parent-a', 'parent-b'])
        with patch('app.rag.pipeline.AsyncSessionFactory') as factory:
            factory.return_value.__aenter__.return_value = session
            for children, expected in [
                ([], False),
                ([{'parent_id': 'parent-a'}], False),
                ([{'parent_id': 'parent-a'}, {'parent_id': 'parent-b'}], True),
                ([{'parent_id': 'parent-a'}, {'parent_id': 'parent-b'}, {'parent_id': 'stale'}], False),
            ]:
                with self.subTest(children=children):
                    self.pipeline._vector_store.client.query.return_value = children
                    self.assertEqual(await self.pipeline.is_ingested(12), expected)

    async def test_embedding_requests_are_small_batches(self):
        pipeline = object.__new__(RAGPipeline)
        pipeline._vector_store = Mock()
        pipeline._vector_store.collection_name = 'test_collection'
        pipeline._vector_store.client.has_collection.return_value = True
        chunks = list(range(35))
        pipeline._save_child_chunks(12, chunks)
        batches = [call.args[0] for call in pipeline._vector_store.add_documents.call_args_list]
        self.assertEqual([len(batch) for batch in batches], [16, 16, 3])
        self.assertEqual([item for batch in batches for item in batch], chunks)
        pipeline._vector_store.delete.assert_called_once_with(expr='product_id == 12')

    async def test_import_script_skips_complete_and_continues_after_failure(self):
        from scripts import ingest_rag
        products = [SimpleNamespace(id=i, name=str(i)) for i in range(3)]
        pipeline = Mock()
        pipeline.is_ingested = AsyncMock(side_effect=[True, False, False, True])
        pipeline.ingest = AsyncMock(side_effect=[RuntimeError('test failure'), None])
        with patch.object(ingest_rag, 'AsyncSessionFactory'), \
                patch.object(ingest_rag, 'ProductService') as service, \
                patch.object(ingest_rag, 'rag_pipeline', pipeline), \
                patch.object(ingest_rag, 'close_rag'), \
                patch.object(ingest_rag, 'close_database', AsyncMock()):
            service.return_value.list_product = AsyncMock(return_value=products)
            with self.assertRaises(SystemExit) as failure:
                await ingest_rag.main()
            self.assertEqual(failure.exception.code, 1)
        self.assertEqual([call.args[0].id for call in pipeline.ingest.await_args_list], [1, 2])


if __name__ == '__main__':
    unittest.main(verbosity=2)
