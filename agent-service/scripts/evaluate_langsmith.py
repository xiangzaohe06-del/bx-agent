"""读取LangSmith网页题库，在本机运行完整Agent，业务评分使用Dataset绑定的评估器。"""

import argparse
import asyncio
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# 支持从后端目录直接执行python scripts/evaluate_langsmith.py。
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.core.config import APP_ROOT
from langsmith import Client, aevaluate
from langchain_core.tracers.langchain import wait_for_all_tracers
from app.core.logging import configure_logging
from app.evaluation.remote_dataset import (
    RemoteReference, RemoteEvaluationTarget, evaluate_operational,
    remote_fingerprint, validate_remote_examples,
)
from app.evaluation.runner import EvaluationTarget, evaluation_agent, evaluation_metadata


def arguments():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-name", required=True, help="网页Dataset的完整名称")
    parser.add_argument("--check-only", action="store_true", help="仅远端读取与校验，不调用模型、不创建实验")
    parser.add_argument("--case", action="append", dest="case_ids", help="只运行指定case_id，可重复传入")
    parser.add_argument("--limit", type=int, help="按case_id排序后最多运行多少题")
    parser.add_argument("--timeout", type=int, default=180, help="单题全部轮次的超时秒数")
    parser.add_argument("--repetitions", type=int, default=1, help="每题重复次数，默认1")
    parser.add_argument("--experiment-prefix", default="insurance-web-baseline")
    return parser.parse_args()


async def main(args):
    if args.timeout <= 0 or args.repetitions <= 0 or (args.limit is not None and args.limit <= 0):
        raise ValueError("timeout、repetitions和limit必须为正整数")
    if not os.environ.get("LANGSMITH_API_KEY"):
        raise ValueError("后端.env缺少LANGSMITH_API_KEY")
    configure_logging("INFO")
    client = Client(timeout_ms=15000)
    try:
        dataset = await asyncio.to_thread(client.read_dataset, dataset_name=args.dataset_name)
        snapshot_time = datetime.now(timezone.utc)
        # 一次性读取版本快照；本入口从不创建、同步或覆盖远端题库。
        examples = await asyncio.to_thread(
            lambda: list(client.list_examples(dataset_id=dataset.id, as_of=snapshot_time)))
        validate_remote_examples(examples)
        examples.sort(key=lambda example: example.outputs["case_id"])
        if args.case_ids:
            unknown = set(args.case_ids) - {item.outputs["case_id"] for item in examples}
            if unknown:
                raise ValueError(f"题库不存在case_id：{', '.join(sorted(unknown))}")
            examples = [item for item in examples if item.outputs["case_id"] in args.case_ids]
        if args.limit is not None:
            examples = examples[:args.limit]
        print(f"题库：{dataset.name}，本次选择{len(examples)}题，每题{args.repetitions}次")
        print(f"题库页面：{dataset.url}")
        for example in examples:
            reference = RemoteReference.model_validate(example.outputs)
            print(f"  {reference.case_id} [{reference.category}]")
        if args.check_only:
            print("远端题库校验通过；未初始化Agent、未调用模型、未创建实验。")
            return

        metadata = evaluation_metadata(remote_fingerprint(examples))
        metadata.update({"evaluation_type": "web_dataset", "dataset_snapshot_at": snapshot_time.isoformat(),
                         "semantic_scoring": "dataset_bound_server_evaluators",
                         "integration_sha256": hashlib.sha256(
                             (APP_ROOT / "app/evaluation/remote_dataset.py").read_bytes()
                             + Path(__file__).read_bytes()
                             + (APP_ROOT / "app/evaluation/runner.py").read_bytes()).hexdigest()})
        rows = []
        async with evaluation_agent() as agent:
            target = RemoteEvaluationTarget(EvaluationTarget(agent, timeout_seconds=args.timeout))
            experiment = await aevaluate(
                target.run, data=examples, evaluators=[evaluate_operational], client=client,
                experiment_prefix=args.experiment_prefix, metadata=metadata,
                description="LangSmith网页题库驱动本机完整Agent；语义评分由Dataset关联评估器完成",
                max_concurrency=1, num_repetitions=args.repetitions, blocking=False,
                disable_evaluator_tracing=True,
            )
            async for row in experiment:
                reference = row["example"].outputs
                scores = [{"key": score.key, "score": score.score, "comment": score.comment}
                          for score in row["evaluation_results"]["results"]]
                run = row.get("run")
                output = (run.outputs or {}) if run else {}
                # 不把“运行成功”标为“业务通过”；服务端语义分数可能稍后才到达。
                runtime_ok = (len(scores) == 2 and all(score["score"] == 1 for score in scores)
                              and not row.get("error"))
                rows.append({"case_id": reference["case_id"], "category": reference["category"],
                             "run_id": str(run.id) if run else None,
                             "runtime_ok": runtime_ok, "outputs": output, "operational_scores": scores})
                print(f"{'运行完成' if runtime_ok else '运行异常'}：{reference['case_id']}")
            await experiment.wait()

        report = {"experiment_name": experiment.experiment_name,
                  "experiment_url": await experiment.get_comparison_url(),
                  "dataset_name": dataset.name, "dataset_url": dataset.url, "metadata": metadata,
                  "runtime_ok": sum(row["runtime_ok"] for row in rows), "total": len(rows), "cases": rows,
                  "semantic_scores_status": "请在LangSmith实验页查看；本地未等待服务端业务评估器完成",
                  "note": "运行完成率不是业务正确率；token仅包含已采集的聊天模型用量。"}
        directory = APP_ROOT / "data/evaluation/results"
        directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        path = directory / f"{timestamp}-web.json"
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"运行完成：{report['runtime_ok']}/{report['total']}")
        print(f"实验页面：{report['experiment_url']}")
        print(f"本地报告：{path}")
        print(report["semantic_scores_status"])
        if report["runtime_ok"] != report["total"]:
            raise SystemExit(1)
    finally:
        await asyncio.to_thread(wait_for_all_tracers)
        await asyncio.to_thread(client.flush, timeout=15)
        client.close()


if __name__ == "__main__":
    options = arguments()
    asyncio.run(main(options), loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None)
