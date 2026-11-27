import hashlib
import json
from pathlib import Path
from uuid import uuid5, NAMESPACE_URL

from langsmith.utils import LangSmithNotFoundError
from pydantic import BaseModel, ConfigDict, Field, model_validator


class CaseInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[str] = Field(min_length=1, max_length=10)

    @model_validator(mode="after")
    def validate_messages(self):
        if any(not message.strip() or len(message) > 2000 for message in self.messages):
            raise ValueError("测评消息不能为空，每条最多2000字符")
        return self


class ExpectedChecks(BaseModel):
    """明确列出规则检查范围，不将规则通过当作完整语义正确。"""
    model_config = ConfigDict(extra="forbid")
    workflow: str | None = None
    min_delegations: int = Field(default=0, ge=0)
    max_delegations: int = Field(default=3, ge=0)
    answer_patterns: list[str] = Field(default_factory=list)
    any_terms: list[str] = Field(default_factory=list)
    forbidden_terms: list[str] = Field(default_factory=list)
    number_only: int | None = None
    required_tool: str | None = None
    product_id: int | None = None
    evidence_patterns: list[str] = Field(default_factory=list)
    require_sources: bool = False


class EvaluationCase(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    title: str
    category: str
    inputs: CaseInputs
    expected: ExpectedChecks


class EvaluationDataset(BaseModel):
    model_config = ConfigDict(extra="forbid")
    name: str
    version: str
    description: str
    cases: list[EvaluationCase] = Field(min_length=1)

    @model_validator(mode="after")
    def validate_case_ids(self):
        if len({case.case_id for case in self.cases}) != len(self.cases):
            raise ValueError("测评用例case_id不能重复")
        return self


def load_dataset(path: Path):
    return EvaluationDataset.model_validate_json(path.read_text(encoding="utf-8"))


def dataset_fingerprint(dataset):
    content = json.dumps(dataset.model_dump(), ensure_ascii=False, sort_keys=True)
    return hashlib.sha256(content.encode("utf-8")).hexdigest()


def prepare_dataset(client, dataset):
    """同一题库重复执行不重复插入；发现远端被修改时停止，避免无意覆盖。"""
    fingerprint = dataset_fingerprint(dataset)
    try:
        remote = client.read_dataset(dataset_name=dataset.name)
        if (remote.metadata or {}).get("dataset_sha256") != fingerprint:
            raise ValueError("远端同名题库与本地版本不一致，请更换题库名称或检查远端内容")
    except LangSmithNotFoundError:
        remote = client.create_dataset(
            dataset_name=dataset.name, description=dataset.description,
            metadata={"dataset_sha256": fingerprint, "version": dataset.version, "source": "bx-agent"},
        )

    expected = {}
    for case in dataset.cases:
        example_id = str(uuid5(NAMESPACE_URL, f"{remote.id}:{case.case_id}"))
        expected[example_id] = {
            "id": example_id,
            "inputs": case.inputs.model_dump(),
            "outputs": case.expected.model_dump(),
            "metadata": {"case_id": case.case_id, "title": case.title, "category": case.category},
        }
    existing = list(client.list_examples(dataset_id=remote.id))
    for example in existing:
        item = expected.get(str(example.id))
        if item is None or example.inputs != item["inputs"] or example.outputs != item["outputs"]:
            raise ValueError("远端用例已改变，请使用新名称建立新版本，不覆盖已有测评记录")
    existing_ids = {str(example.id) for example in existing}
    missing = [item for key, item in expected.items() if key not in existing_ids]
    if missing:
        client.create_examples(dataset_id=remote.id, examples=missing)
    examples = list(client.list_examples(dataset_id=remote.id))
    by_id = {str(example.id): example for example in examples}
    return remote, [by_id[key] for key in expected]
