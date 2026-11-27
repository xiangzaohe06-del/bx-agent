"""读取网页维护的题库；参考答案只交给评分器，不传入智能体。"""

import hashlib
import json

from pydantic import BaseModel, ConfigDict, Field, field_validator

from .dataset import CaseInputs


class RemoteInputs(BaseModel):
    model_config = ConfigDict(extra="forbid")
    messages: list[str]

    @field_validator("messages", mode="before")
    @classmethod
    def decode_messages(cls, value):
        # CSV中的JSON数组通常被导入为字符串，也兼容网页直接维护的数组。
        if isinstance(value, str):
            try:
                value = json.loads(value)
            except ValueError as exc:
                raise ValueError("messages必须是JSON字符串数组") from exc
        return CaseInputs(messages=value).messages


class RemoteReference(BaseModel):
    model_config = ConfigDict(extra="forbid")
    case_id: str = Field(pattern=r"^[a-z][a-z0-9_-]*$")
    category: str = Field(min_length=1)
    expected_points: str = Field(min_length=1)
    forbidden_behaviors: str = Field(min_length=1)
    reference_evidence: str = ""


def validate_remote_examples(examples):
    if not examples:
        raise ValueError("远端题库为空")
    seen = set()
    for example in examples:
        RemoteInputs.model_validate(example.inputs)
        reference = RemoteReference.model_validate(example.outputs or {})
        if reference.case_id in seen:
            raise ValueError(f"重复的case_id：{reference.case_id}")
        seen.add(reference.case_id)


def remote_fingerprint(examples):
    # 固定本次取到的题目快照；网页后续编辑不改变已经加载的实验内容。
    content = [{"id": str(item.id), "inputs": item.inputs, "outputs": item.outputs}
               for item in sorted(examples, key=lambda item: str(item.id))]
    return hashlib.sha256(json.dumps(content, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def make_evaluation_record(messages, outputs):
    """提供唯一、清晰的评分入口，避免评委从内部状态中误选子Agent草稿。"""
    turns = outputs.get("turns", [])
    return {
        "error": outputs.get("error"),
        "execution_complete": (not outputs.get("error") and len(turns) == len(messages)
                               and all(turn.get("done") and turn.get("answer", "").strip()
                                       for turn in turns)),
        "conversation": [{"turn": index + 1, "user": message,
                          "assistant": turns[index].get("answer", "") if index < len(turns) else "",
                          "tools": turns[index].get("tools", []) if index < len(turns) else [],
                          "done": turns[index].get("done", False) if index < len(turns) else False}
                         for index, message in enumerate(messages)],
        "evidence_note": "tools为各轮真实工具回调；output_truncated=true表示证据被截断，不能据此断言原工具未返回某事实。",
    }


class RemoteEvaluationTarget:
    def __init__(self, target):
        self.target = target

    async def run(self, inputs):
        # 只放行用户消息；category、标准答案、条款参考等均不能传入被测系统。
        messages = RemoteInputs.model_validate(inputs).messages
        outputs = await self.target.run({"messages": messages})
        outputs["final_answer"] = outputs["turns"][-1]["answer"] if outputs.get("turns") else ""
        outputs["evaluation_record"] = make_evaluation_record(messages, outputs)
        return outputs


def evaluate_operational(inputs, outputs, reference_outputs):
    """本地仅做运行检查；业务语义质量由Dataset上绑定的评估器评分。"""
    from .evaluators import feedback

    messages = RemoteInputs.model_validate(inputs).messages
    record = make_evaluation_record(messages, outputs or {})
    return {"results": [
        feedback("execution_complete", record["execution_complete"], "所有输入轮次正常结束且回复非空"),
        feedback("cleanup_complete", (outputs or {}).get("cleanup_ok", False), "临时会话与检查点已清理"),
    ]}
