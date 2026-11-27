import re

from .dataset import ExpectedChecks


def feedback(key, passed, description):
    return {"key": key, "score": int(bool(passed)), "comment": description}


def evaluate_rules(inputs, outputs, reference_outputs):
    """不调用模型的规则评分；每项失败都给出可以复核的检查范围。"""
    expected = ExpectedChecks.model_validate(reference_outputs)
    outputs = outputs or {}
    turns = outputs.get("turns", [])
    last = turns[-1] if turns else {}
    answer = last.get("answer", "")
    complete = (not outputs.get("error") and len(turns) == len(inputs["messages"])
                and all(turn.get("done") and turn.get("answer", "").strip() for turn in turns))
    scores = [feedback("execution_complete", complete, "所有轮次正常结束，并返回非空回复")]

    matches = all(re.search(pattern, answer, re.S) for pattern in expected.answer_patterns)
    matches = matches and (not expected.any_terms or any(term in answer for term in expected.any_terms))
    matches = matches and not any(term in answer for term in expected.forbidden_terms)
    if expected.number_only is not None:
        # 测试问题明确要求只回复阿拉伯数字，避免旧预算与新预算同时出现也被判对。
        cleaned = re.sub(r"[\s*`，,。]", "", answer)
        matches = matches and cleaned == str(expected.number_only)
    scores.append(feedback("response_rules", complete and matches,
                           "仅检查题库规定的表达、数字和禁止项，不代表完整语义正确"))

    workflow_ok = not expected.workflow or last.get("workflow") == expected.workflow
    workflow_ok = workflow_ok and all(
        0 <= turn.get("delegation_count", 0) <= expected.max_delegations for turn in turns
    ) and last.get("delegation_count", 0) >= expected.min_delegations
    workers = last.get("workers", [])
    workflow_ok = workflow_ok and (not workers or workers[-1].get("status") != "failed")
    scores.append(feedback("workflow_rules", complete and workflow_ok,
                           "检查最终工作流、委派次数上限和子任务是否失败"))

    if expected.required_tool:
        tools = [tool for tool in outputs.get("tools", []) if tool["name"] == expected.required_tool
                 and not tool.get("error")]
        if expected.product_id is not None:
            tools = [tool for tool in tools if str(tool.get("arguments", {}).get("product_id"))
                     == str(expected.product_id)]
        evidence_ok = any(all(re.search(pattern, tool.get("output", ""), re.S)
                              for pattern in expected.evidence_patterns) for tool in tools)
        if expected.require_sources:
            evidence_ok = evidence_ok and any(worker.get("sources") for worker in workers)
        scores.append(feedback("tool_evidence", complete and evidence_ok,
                               "实际工具调用、产品参数、返回原文与来源字段必须同时符合要求"))

    internal_markers = ["本轮内部任务数据", "以下为较早对话的摘要", "SupervisorDecision", "WorkerResult"]
    clean = not any(marker in turn.get("answer", "") for turn in turns for marker in internal_markers)
    scores.append(feedback("public_reply_clean", complete and clean, "公开回复不包含已知内部数据标记"))
    scores.append(feedback("cleanup_complete", outputs.get("cleanup_ok", False), "本次临时会话与检查点已清理"))
    return {"results": scores}
