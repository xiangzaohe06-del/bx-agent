import copy
import pathlib
import sys
import unittest
from types import SimpleNamespace
from uuid import uuid4

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from langsmith.utils import LangSmithNotFoundError

from app.core.config import APP_ROOT
from app.evaluation.dataset import load_dataset, prepare_dataset
from app.evaluation.evaluators import evaluate_rules
from app.evaluation.runner import EvaluationTraceCollector


class DatasetClient:
    def __init__(self):
        self.dataset = None
        self.examples = []
        self.created_batches = 0

    def read_dataset(self, **kwargs):
        if self.dataset is None:
            raise LangSmithNotFoundError("not found")
        return self.dataset

    def create_dataset(self, dataset_name, description, metadata):
        self.dataset = SimpleNamespace(id=uuid4(), name=dataset_name, metadata=metadata)
        return self.dataset

    def list_examples(self, **kwargs):
        return self.examples

    def create_examples(self, dataset_id, examples):
        self.created_batches += 1
        self.examples.extend(SimpleNamespace(**example) for example in examples)


class EvaluationTests(unittest.TestCase):
    def setUp(self):
        self.dataset = load_dataset(APP_ROOT / "data/evaluation/insurance_baseline_v1.json")

    def scores(self, case, outputs):
        return {score["key"]: score["score"] for score in evaluate_rules(
            case.inputs.model_dump(), outputs, case.expected.model_dump()
        )["results"]}

    def rag_output(self):
        return {
            "error": None, "cleanup_ok": True,
            "turns": [{"answer": "重大疾病等待期为180日，依据2.3.1等待期条款。", "done": True,
                       "workflow": "recommendation_plan", "delegation_count": 1,
                       "workers": [{"status": "completed", "sources": ["2.3.1 等待期"]}]}],
            "tools": [{"name": "retrieve_insurance_clause", "arguments": {"product_id": 2},
                       "error": False, "output": "section_path: 2.3.1 等待期\n本合同生效日起 180 日为等待期。"}],
        }

    def test_dataset_creation_is_idempotent(self):
        client = DatasetClient()
        remote, examples = prepare_dataset(client, self.dataset)
        second, repeated = prepare_dataset(client, self.dataset)
        self.assertEqual(remote.id, second.id)
        self.assertEqual([example.id for example in examples], [example.id for example in repeated])
        self.assertEqual(client.created_batches, 1)
        self.assertEqual(len(examples), 5)

    def test_remote_dataset_changes_are_not_overwritten(self):
        client = DatasetClient()
        prepare_dataset(client, self.dataset)
        client.examples[0].outputs = {"changed": True}
        with self.assertRaises(ValueError):
            prepare_dataset(client, self.dataset)
        self.assertEqual(client.created_batches, 1)

    def test_wrong_product_does_not_pass_despite_correct_number(self):
        case = self.dataset.cases[-1]
        output = self.rag_output()
        self.assertEqual(self.scores(case, output)["tool_evidence"], 1)
        output["tools"][0]["arguments"]["product_id"] = 3
        self.assertEqual(self.scores(case, output)["response_rules"], 1)
        self.assertEqual(self.scores(case, output)["tool_evidence"], 0)

    def test_other_180_day_clause_or_invented_sources_do_not_pass(self):
        case = self.dataset.cases[-1]
        output = self.rag_output()
        output["tools"][0]["output"] = "重大疾病多次给付间隔180日"
        self.assertEqual(self.scores(case, output)["tool_evidence"], 0)
        output["tools"] = []
        self.assertEqual(self.scores(case, output)["tool_evidence"], 0)

    def test_failed_execution_or_missing_cleanup_cannot_be_all_passed(self):
        case = self.dataset.cases[-1]
        output = self.rag_output()
        output["error"] = "TimeoutError"
        self.assertEqual(self.scores(case, output)["execution_complete"], 0)
        output = self.rag_output()
        output["cleanup_ok"] = False
        self.assertEqual(self.scores(case, output)["cleanup_complete"], 0)

    def test_old_and_new_budget_together_does_not_pass_number_only_requirement(self):
        case = self.dataset.cases[2]
        turn = {"answer": "3000改成5000", "done": True, "workflow": "recommendation_plan", "delegation_count": 1}
        output = {"error": None, "cleanup_ok": True, "turns": [copy.deepcopy(turn), turn]}
        self.assertEqual(self.scores(case, output)["response_rules"], 0)
        turn["answer"] = "**5,000**"
        self.assertEqual(self.scores(case, output)["response_rules"], 1)

    def test_trace_usage_does_not_count_same_call_twice_or_hide_unknown_usage(self):
        collector = EvaluationTraceCollector()
        first, second = uuid4(), uuid4()
        collector.on_chat_model_start({}, [], run_id=first)
        collector.on_chat_model_start({}, [], run_id=second)
        response = SimpleNamespace(generations=[[SimpleNamespace(message=SimpleNamespace(
            usage_metadata={"input_tokens": 100, "output_tokens": 20}
        ))]], llm_output=None)
        collector.on_llm_end(response, run_id=first)
        collector.on_llm_end(response, run_id=first)
        self.assertEqual(collector.usage()["known_input_tokens"], 100)
        self.assertEqual(collector.usage()["calls_without_usage"], 1)

    def test_tool_arguments_use_original_inputs_instead_of_python_dictionary_text(self):
        collector = EvaluationTraceCollector()
        run_id = uuid4()
        collector.on_tool_start(
            {"name": "retrieve_insurance_clause"}, "{'product_id': 2}",
            run_id=run_id, inputs={"product_id": 2, "k": 5},
        )
        self.assertEqual(collector.tools[str(run_id)]["arguments"]["product_id"], 2)


if __name__ == '__main__':
    unittest.main(verbosity=2)
