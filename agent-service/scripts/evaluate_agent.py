import argparse
import asyncio
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

from app.core.config import APP_ROOT

from langsmith import Client, aevaluate
from langchain_core.tracers.langchain import wait_for_all_tracers
from app.core.logging import configure_logging
from app.evaluation.dataset import load_dataset, prepare_dataset, dataset_fingerprint
from app.evaluation.evaluators import evaluate_rules
from app.evaluation.runner import EvaluationTarget, evaluation_agent, evaluation_metadata


def arguments():
    parser = argparse.ArgumentParser(description="本机运行客服测评，结果上传LangSmith，不需要启动前端或后端HTTP服务")
    parser.add_argument("--dataset", type=Path, default=APP_ROOT / "data/evaluation/insurance_baseline_v1.json")
    parser.add_argument("--prepare-only", action="store_true", help="只创建或检查题库，不调用模型")
    parser.add_argument("--case", action="append", dest="case_ids", help="只执行指定case_id，可重复传入")
    parser.add_argument("--timeout", type=int, default=180, help="每个用例超时秒数，默认180")
    return parser.parse_args()


async def main(args):
    configure_logging("INFO")
    dataset = load_dataset(args.dataset)
    if args.timeout <= 0:
        raise ValueError("超时秒数必须大于0")
    if args.case_ids and set(args.case_ids) - {case.case_id for case in dataset.cases}:
        raise ValueError("指定的case_id不存在，请检查本地题库")
    if not os.environ.get("LANGSMITH_API_KEY"):
        raise ValueError("请先在后端.env中配置LANGSMITH_API_KEY")

    client = Client(timeout_ms=15000)
    try:
        remote, examples = await asyncio.to_thread(prepare_dataset, client, dataset)
        print(f"题库：{remote.name}，共{len(examples)}个用例")
        print(f"题库页面：{remote.url}")
        if args.prepare_only:
            print("题库已准备完成，本次未调用模型。")
            return
        if args.case_ids:
            examples = [example for example in examples if example.metadata["case_id"] in args.case_ids]

        metadata = evaluation_metadata(dataset_fingerprint(dataset))
        rows = []
        async with evaluation_agent() as agent:
            target = EvaluationTarget(agent, timeout_seconds=args.timeout)
            # 单并发、每例一次；不使用模型评委，也不让测评触发PDF导入。
            experiment = await aevaluate(
                target.run, data=examples, evaluators=[evaluate_rules], client=client,
                experiment_prefix="insurance-baseline", metadata=metadata,
                description="本机真实客服工作流的规则基线；业务回答仍需人工复核",
                max_concurrency=1, num_repetitions=1, blocking=False,
                disable_evaluator_tracing=True,
            )
            async for row in experiment:
                example = row["example"]
                results = row["evaluation_results"]["results"]
                scores = [{"key": score.key, "score": score.score, "comment": score.comment} for score in results]
                passed = bool(scores) and all(score["score"] == 1 for score in scores)
                print(f"{'通过' if passed else '未通过'}：{example.metadata['title']}")
                rows.append({
                    "case_id": example.metadata["case_id"], "title": example.metadata["title"],
                    "run_id": str(row["run"].id), "passed": passed,
                    "outputs": row["run"].outputs, "scores": scores,
                })
            await experiment.wait()

        report = {
            "experiment_name": experiment.experiment_name,
            "experiment_url": await experiment.get_comparison_url(),
            "dataset_name": dataset.name, "dataset_url": remote.url, "metadata": metadata,
            "passed": sum(row["passed"] for row in rows), "total": len(rows), "cases": rows,
            "note": "规则基线，不代表完整语义正确率；聊天token统计不包括原生向量SDK等未采集用量。",
        }
        report_dir = APP_ROOT / "data/evaluation/results"
        report_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        report_path = report_dir / f"{timestamp}.json"
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"测评完成：{report['passed']}/{report['total']}通过")
        print(f"实验页面：{report['experiment_url']}")
        print(f"本地报告：{report_path}")
        if report["passed"] != report["total"]:
            raise SystemExit(1)
    finally:
        await asyncio.to_thread(wait_for_all_tracers)
        await asyncio.to_thread(client.flush, timeout=15)
        client.close()


if __name__ == "__main__":
    options = arguments()
    asyncio.run(main(options), loop_factory=asyncio.SelectorEventLoop if sys.platform == "win32" else None)
