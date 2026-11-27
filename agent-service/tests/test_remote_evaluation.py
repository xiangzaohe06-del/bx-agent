import asyncio
import csv
import json
import pathlib
import sys
import unittest
from types import SimpleNamespace

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from app.core.config import APP_ROOT
from app.evaluation.remote_dataset import (
    RemoteInputs, RemoteEvaluationTarget, evaluate_operational,
    remote_fingerprint, validate_remote_examples,
)


class RemoteEvaluationTests(unittest.TestCase):
    def examples(self):
        with (APP_ROOT / 'data/evaluation/insurance_langsmith_v1.csv').open(
                encoding='utf-8-sig', newline='') as stream:
            return [SimpleNamespace(id=str(index), inputs={'messages': row.pop('messages')}, outputs=row)
                    for index, row in enumerate(csv.DictReader(stream))]

    def test_csv_round_trip_and_all_reference_fields_are_separate(self):
        examples = self.examples()
        validate_remote_examples(examples)
        self.assertEqual(len(examples), 10)
        self.assertEqual(len({item.outputs['category'] for item in examples}), 5)
        case = next(item for item in examples if item.outputs['case_id'] == 'context_budget')
        messages = RemoteInputs.model_validate(case.inputs).messages
        self.assertEqual(len(messages), 2)
        self.assertIn('3000', messages[-1])
        self.assertEqual(set(case.inputs), {'messages'})

    def test_bad_json_and_reference_leakage_are_rejected(self):
        for value in ('not json', '{"messages": []}', '[]', '[" "]', '[{"role":"user"}]'):
            with self.assertRaises(ValueError):
                RemoteInputs.model_validate({'messages': value})
        with self.assertRaises(ValueError):
            RemoteInputs.model_validate({'messages': ['你好'], 'expected_points': '标准答案'})

    def test_duplicate_ids_and_missing_references_are_rejected(self):
        examples = self.examples()
        examples[1].outputs['case_id'] = examples[0].outputs['case_id']
        with self.assertRaises(ValueError):
            validate_remote_examples(examples)
        examples = self.examples()
        del examples[0].outputs['expected_points']
        with self.assertRaises(ValueError):
            validate_remote_examples(examples)

    def test_target_forwards_only_decoded_messages_and_keeps_tools_in_correct_turn(self):
        class FakeTarget:
            async def run(self, inputs):
                self.received = inputs
                return {'turns': [{'answer': '首轮回复', 'done': True, 'tools': []},
                                  {'answer': '3000元', 'done': True,
                                   'tools': [{'name': 'retrieve_insurance_clause', 'output': '证据'}]}],
                        'error': None, 'cleanup_ok': True}

        target = FakeTarget()
        wrapped = RemoteEvaluationTarget(target)
        inputs = {'messages': json.dumps(['先记下5000', '改为3000'])}
        output = asyncio.run(wrapped.run(inputs))
        self.assertEqual(target.received, {'messages': ['先记下5000', '改为3000']})
        self.assertEqual(output['final_answer'], '3000元')
        self.assertEqual(output['evaluation_record']['conversation'][0]['tools'], [])
        self.assertEqual(len(output['evaluation_record']['conversation'][1]['tools']), 1)
        self.assertTrue(output['evaluation_record']['execution_complete'])

    def test_incomplete_run_never_passes_execution_check(self):
        scores = evaluate_operational({'messages': ['问题1', '问题2']}, {
            'turns': [{'answer': '回复', 'done': True}], 'cleanup_ok': True, 'error': None,
        }, {})['results']
        self.assertEqual(scores[0]['score'], 0)
        self.assertEqual(scores[1]['score'], 1)

    def test_fingerprint_detects_reference_changes_and_ignores_order(self):
        examples = self.examples()
        original = remote_fingerprint(examples)
        self.assertEqual(original, remote_fingerprint(list(reversed(examples))))
        examples[0].outputs['expected_points'] += '新增要求'
        self.assertNotEqual(original, remote_fingerprint(examples))


if __name__ == '__main__':
    unittest.main(verbosity=2)
