import pathlib
import sys
import unittest
from unittest.mock import AsyncMock

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / 'agent-service'))
from langchain_core.documents import Document
from app.infra.reranker import ChatModelReranker, RerankResult


class RerankerTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        self.documents = [Document(page_content='clause ' + str(i), metadata={'parent_id': str(i)}) for i in range(3)]
        self.ranker = object.__new__(ChatModelReranker)
        self.ranker._model = AsyncMock()

    async def test_rank_preserves_original_text_and_metadata(self):
        self.ranker._model.ainvoke.return_value = RerankResult(indices=[2, 0, 1])
        ranked = await self.ranker.rerank('question', self.documents, 2)
        self.assertEqual(ranked, [self.documents[2], self.documents[0]])
        self.assertIs(ranked[0], self.documents[2])
        self.assertEqual([d.page_content for d in self.documents], ['clause 0', 'clause 1', 'clause 2'])

    async def test_model_outage_keeps_retrieval_available(self):
        self.ranker._model.ainvoke.side_effect = TimeoutError('test outage')
        self.assertEqual(await self.ranker.rerank('question', self.documents, 2), self.documents[:2])

    async def test_invalid_indices_never_index_untrusted_documents(self):
        for indices in [[0, 0], [-1], [3], []]:
            with self.subTest(indices=indices):
                self.ranker._model.ainvoke.return_value = RerankResult(indices=indices)
                self.assertEqual(await self.ranker.rerank('question', self.documents, 2), self.documents[:2])

    async def test_incomplete_ranking_is_completed_without_duplicates(self):
        self.ranker._model.ainvoke.return_value = RerankResult(indices=[2])
        self.assertEqual(await self.ranker.rerank('question', self.documents, 3), [self.documents[2], self.documents[0], self.documents[1]])

    async def test_empty_single_and_zero_k_do_not_spend_tokens(self):
        self.assertEqual(await self.ranker.rerank('question', [], 3), [])
        self.assertEqual(await self.ranker.rerank('question', self.documents, 0), [])
        self.assertEqual(await self.ranker.rerank('question', self.documents[:1], 3), self.documents[:1])
        self.ranker._model.ainvoke.assert_not_called()


if __name__ == '__main__':
    unittest.main(verbosity=2)
