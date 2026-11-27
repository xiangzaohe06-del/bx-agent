import pathlib
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from app.infra.embeddings import BailianVLTextEmbeddings


def success(batch):
    return SimpleNamespace(
        status_code=200, code='',
        output={'embeddings': [
            {'index': i, 'embedding': [float(int(item['text']))] * 256}
            for i, item in reversed(list(enumerate(batch)))
        ]},
    )


class CloudEmbeddingTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.model = BailianVLTextEmbeddings(dimensions=256)

    async def test_batch_limit_and_response_order(self):
        with patch('app.infra.embeddings.MultiModalEmbedding.call', side_effect=lambda **kw: success(kw['input'])) as call:
            vectors = self.model.embed_documents([str(i) for i in range(45)])
        self.assertEqual([len(item.kwargs['input']) for item in call.call_args_list], [20, 20, 5])
        self.assertEqual([v[0] for v in vectors], list(range(45)))
        self.assertTrue(all(item.kwargs['enable_fusion'] is False for item in call.call_args_list))
        self.assertTrue(all(item.kwargs['dimension'] == 256 for item in call.call_args_list))

    async def test_async_query_uses_same_model(self):
        with patch('app.infra.embeddings.MultiModalEmbedding.call', side_effect=lambda **kw: success(kw['input'])) as call:
            vector = await self.model.aembed_query('7')
        self.assertEqual(vector, [7.0] * 256)
        self.assertEqual(call.call_args.kwargs['model'], 'qwen3-vl-embedding')

    async def test_empty_batch_costs_no_request_and_blank_text_rejected(self):
        with patch('app.infra.embeddings.MultiModalEmbedding.call') as call:
            self.assertEqual(self.model.embed_documents([]), [])
            with self.assertRaises(ValueError):
                self.model.embed_documents([' '])
            call.assert_not_called()

    async def test_api_error_does_not_return_fake_vectors(self):
        response = SimpleNamespace(status_code=401, code='InvalidApiKey')
        with patch('app.infra.embeddings.MultiModalEmbedding.call', return_value=response):
            with self.assertRaisesRegex(RuntimeError, '401'):
                self.model.embed_documents(['1'])

    async def test_missing_duplicate_or_wrong_dimension_rejected(self):
        bad_items = [
            [],
            [{'index': 0, 'embedding': [1.0] * 256}] * 2,
            [{'index': 0, 'embedding': [1.0] * 256}, {'index': 1, 'embedding': [1.0]}],
            [{'index': 0, 'embedding': [float('nan')] * 256}, {'index': 1, 'embedding': [1.0] * 256}],
        ]
        for items in bad_items:
            with self.subTest(items_count=len(items)):
                response = SimpleNamespace(status_code=200, output={'embeddings': items})
                with patch('app.infra.embeddings.MultiModalEmbedding.call', return_value=response):
                    with self.assertRaises(RuntimeError):
                        self.model.embed_documents(['1', '2'])

    async def test_unsupported_model_or_dimensions_rejected(self):
        with self.assertRaises(ValueError):
            BailianVLTextEmbeddings(model='qwen4-max')
        with self.assertRaises(ValueError):
            BailianVLTextEmbeddings(dimensions=100)


if __name__ == '__main__':
    unittest.main(verbosity=2)
