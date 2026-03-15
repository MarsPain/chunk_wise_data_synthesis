from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import logging
from pathlib import Path
import re
import sys
from typing import Any


def setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        datefmt="%H:%M:%S",
    )


logger = logging.getLogger(__name__)


def _ensure_src_on_path() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


_ensure_src_on_path()

from backends import OpenAIBackendConfig, OpenAILLMModel
from backends.openai import DEFAULT_BASE_URL, DEFAULT_MODEL
from generation_types import GenerationConfig, GenerationPlan
from model import LLMRequest
from pipelines import ChunkWiseGenerationPipeline
from quality import compare_chunked_vs_one_shot, evaluate_generation_coherence
from tokenization import WhitespaceTokenizer


@dataclass(frozen=True)
class EvalCase:
    case_id: str
    topic: str
    objective: str
    target_tokens: int
    audience: str
    tone: str
    manual_plan_path: Path | None = None


def _sanitize_filename(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value.strip())
    return cleaned.strip("_") or "case"


def _load_manual_plan(path: Path) -> GenerationPlan:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"manual plan at {path} must be a JSON object")
    return GenerationPlan.from_dict(payload)


def _load_cases(path: Path) -> list[EvalCase]:
    payload: Any = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, dict):
        raw_cases = payload.get("cases")
    else:
        raw_cases = payload
    if not isinstance(raw_cases, list):
        raise ValueError("cases file must be a list or an object with `cases` list")

    cases: list[EvalCase] = []
    root = path.parent
    for index, item in enumerate(raw_cases):
        if not isinstance(item, dict):
            raise ValueError(f"case[{index}] must be a JSON object")
        case_id = str(item.get("id", "")).strip() or f"case-{index + 1}"
        manual_plan_path_raw = str(item.get("manual_plan_path", "")).strip()
        manual_plan_path = None
        if manual_plan_path_raw:
            candidate = Path(manual_plan_path_raw)
            manual_plan_path = candidate if candidate.is_absolute() else root / candidate

        case = EvalCase(
            case_id=case_id,
            topic=str(item.get("topic", "")).strip(),
            objective=str(item.get("objective", "")).strip(),
            target_tokens=int(item.get("target_tokens", 1500) or 1500),
            audience=str(item.get("audience", "")).strip(),
            tone=str(item.get("tone", "")).strip(),
            manual_plan_path=manual_plan_path,
        )
        if case.manual_plan_path is None and (not case.topic or not case.objective):
            raise ValueError(
                f"case[{index}] must include topic and objective when manual_plan_path is absent"
            )
        cases.append(case)

    return cases


def _render_one_shot_prompt(plan: GenerationPlan, prompt_language: str) -> str:
    plan_json = json.dumps(plan.to_dict(), ensure_ascii=False, indent=2)
    if prompt_language == "zh":
        return "\n\n".join(
            [
                "你将一次性生成完整长文。",
                "必须使用二级标题并严格遵循下列章节顺序：",
                *[f"## {section.title}" for section in plan.sections],
                "每个章节需覆盖该章节 key_points 和 required_entities，不要提前展开后续章节的核心内容。",
                "章节之间使用自然过渡句衔接。",
                "只输出正文，不要解释。",
                "生成计划：",
                plan_json,
            ]
        )

    return "\n\n".join(
        [
            "Generate the full long-form article in one pass.",
            "Use markdown H2 headings and follow this exact section order:",
            *[f"## {section.title}" for section in plan.sections],
            "Each section must cover its own key_points and required_entities.",
            "Do not prematurely expand the core content of later sections.",
            "Use natural transition sentences between sections.",
            "Output only the article body.",
            "Generation plan:",
            plan_json,
        ]
    )


def _normalize_title(value: str) -> str:
    return re.sub(r"\s+", " ", value.strip().lower())


def _split_evenly(text: str, n_parts: int) -> list[str]:
    cleaned = text.strip()
    if n_parts <= 1 or not cleaned:
        return [cleaned]
    chunk_size = max(len(cleaned) // n_parts, 1)
    segments: list[str] = []
    for idx in range(n_parts):
        start = idx * chunk_size
        end = len(cleaned) if idx == n_parts - 1 else (idx + 1) * chunk_size
        segments.append(cleaned[start:end].strip())
    return segments


def _extract_sections_from_one_shot_output(text: str, plan: GenerationPlan) -> list[str]:
    matches = list(re.finditer(r"(?im)^##\s*(.+?)\s*$", text))
    if not matches:
        return _split_evenly(text, len(plan.sections))

    parsed: list[tuple[str, str]] = []
    for idx, match in enumerate(matches):
        heading = match.group(1).strip()
        start = match.end()
        end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        body = text[start:end].strip()
        parsed.append((heading, body))

    section_outputs: list[str] = []
    used = [False] * len(parsed)
    for section in plan.sections:
        title_norm = _normalize_title(section.title)
        selected = ""
        for idx, (heading, body) in enumerate(parsed):
            if used[idx]:
                continue
            heading_norm = _normalize_title(heading)
            if title_norm in heading_norm or heading_norm in title_norm:
                selected = body
                used[idx] = True
                break
        section_outputs.append(selected)

    if any(not part for part in section_outputs):
        fallback = _split_evenly(text, len(plan.sections))
        section_outputs = [
            part if part else fallback[idx]
            for idx, part in enumerate(section_outputs)
        ]

    return section_outputs


def _run_case(
    case: EvalCase,
    pipeline: ChunkWiseGenerationPipeline,
    model: OpenAILLMModel,
    prompt_language: str,
) -> dict[str, Any]:
    logger.info("[AB] Running case: %s", case.case_id)

    if case.manual_plan_path is not None:
        manual_plan = _load_manual_plan(case.manual_plan_path)
        chunked_result = pipeline.run(manual_plan=manual_plan)
    else:
        chunked_result = pipeline.run(
            topic=case.topic,
            objective=case.objective,
            target_tokens=case.target_tokens,
            audience=case.audience,
            tone=case.tone,
        )

    plan = chunked_result.plan
    one_shot_prompt = _render_one_shot_prompt(plan=plan, prompt_language=prompt_language)
    one_shot_text = model.generate(
        LLMRequest(task="section_generation", prompt=one_shot_prompt)
    ).strip()
    one_shot_sections = _extract_sections_from_one_shot_output(
        text=one_shot_text,
        plan=plan,
    )

    chunked_metrics = evaluate_generation_coherence(
        plan=plan,
        section_outputs=chunked_result.section_outputs,
        final_text=chunked_result.final_text,
    )
    one_shot_metrics = evaluate_generation_coherence(
        plan=plan,
        section_outputs=one_shot_sections,
        final_text=one_shot_text,
    )
    comparison = compare_chunked_vs_one_shot(
        case_id=case.case_id,
        chunked_metrics=chunked_metrics,
        one_shot_metrics=one_shot_metrics,
    )

    return {
        "case_id": case.case_id,
        "plan": plan.to_dict(),
        "chunked": {
            "final_text": chunked_result.final_text,
            "section_outputs": chunked_result.section_outputs,
            "metrics": chunked_metrics.to_dict(),
            "quality_report": {
                "coverage_missing": chunked_result.qc_report.coverage_missing,
                "terminology_issues": chunked_result.qc_report.terminology_issues,
                "repetition_issues": chunked_result.qc_report.repetition_issues,
                "drift_issues": chunked_result.qc_report.drift_issues,
                "entity_missing": chunked_result.qc_report.entity_missing,
                "section_warnings": chunked_result.qc_report.section_warnings,
                "consistency_pass_used_fallback": chunked_result.qc_report.consistency_pass_used_fallback,
            },
        },
        "one_shot": {
            "final_text": one_shot_text,
            "section_outputs": one_shot_sections,
            "metrics": one_shot_metrics.to_dict(),
        },
        "comparison": comparison.to_dict(),
    }


def _aggregate_summary(case_results: list[dict[str, Any]]) -> dict[str, Any]:
    if not case_results:
        return {
            "total_cases": 0,
            "chunked_wins": 0,
            "one_shot_wins": 0,
            "ties": 0,
            "avg_chunked_score": 0.0,
            "avg_one_shot_score": 0.0,
            "avg_score_delta": 0.0,
        }

    chunked_scores = [
        float(result["comparison"]["chunked"]["aggregate_score"])
        for result in case_results
    ]
    one_shot_scores = [
        float(result["comparison"]["one_shot"]["aggregate_score"])
        for result in case_results
    ]
    deltas = [float(result["comparison"]["score_delta"]) for result in case_results]
    winners = [str(result["comparison"]["winner"]) for result in case_results]

    return {
        "total_cases": len(case_results),
        "chunked_wins": sum(1 for winner in winners if winner == "chunked"),
        "one_shot_wins": sum(1 for winner in winners if winner == "one_shot"),
        "ties": sum(1 for winner in winners if winner == "tie"),
        "avg_chunked_score": sum(chunked_scores) / len(chunked_scores),
        "avg_one_shot_score": sum(one_shot_scores) / len(one_shot_scores),
        "avg_score_delta": sum(deltas) / len(deltas),
    }


def _write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    summary = payload["summary"]
    rows: list[str] = []
    for result in payload["cases"]:
        comparison = result["comparison"]
        rows.append(
            "| {case_id} | {winner} | {delta:.4f} | {chunked:.4f} | {one_shot:.4f} |  |  |  |".format(
                case_id=result["case_id"],
                winner=comparison["winner"],
                delta=float(comparison["score_delta"]),
                chunked=float(comparison["chunked"]["aggregate_score"]),
                one_shot=float(comparison["one_shot"]["aggregate_score"]),
            )
        )

    content = "\n".join(
        [
            "# Generation A/B Baseline Report",
            "",
            f"- Generated at (UTC): {payload['generated_at_utc']}",
            f"- Cases: {summary['total_cases']}",
            f"- Chunked wins: {summary['chunked_wins']}",
            f"- One-shot wins: {summary['one_shot_wins']}",
            f"- Ties: {summary['ties']}",
            f"- Avg chunked score: {summary['avg_chunked_score']:.4f}",
            f"- Avg one-shot score: {summary['avg_one_shot_score']:.4f}",
            f"- Avg score delta (chunked - one-shot): {summary['avg_score_delta']:.4f}",
            "",
            "## Case Table",
            "",
            "| Case | Auto Winner | Delta | Chunked Score | One-shot Score | Manual Chunked (1-5) | Manual One-shot (1-5) | Notes |",
            "| --- | --- | ---: | ---: | ---: | --- | --- | --- |",
            *rows,
            "",
            "## Manual Review Instructions",
            "",
            "- Manual score should reflect full-document coherence (narrative flow, boundary transitions, global consistency).",
            "- Fill manual scores and notes after reading both outputs for each case.",
        ]
    )
    path.write_text(content, encoding="utf-8")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run one-shot vs chunk-wise generation baseline with coherence metrics."
    )
    parser.add_argument(
        "--cases",
        type=Path,
        default=Path("tests/fixtures/generation_eval_cases.json"),
        help="Path to evaluation cases JSON.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/data/ab_eval_reports"),
        help="Directory for JSON and Markdown reports.",
    )
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--base-url", type=str, default=None)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=16384)
    parser.add_argument("--prefix-window-tokens", type=int, default=1200)
    parser.add_argument(
        "--prompt-language",
        type=str,
        default="en",
        choices=["en", "zh"],
        help="Prompt language for instructions sent to the model.",
    )
    parser.add_argument(
        "--disable-consistency-pass",
        action="store_true",
        help="Disable the final consistency pass for chunk-wise generation.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional max number of cases to run (0 means all).",
    )
    parser.add_argument("--verbose", "-v", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    setup_logging(verbose=args.verbose)

    cases = _load_cases(args.cases)
    if args.limit > 0:
        cases = cases[: args.limit]

    logger.info("Loaded %d cases from %s", len(cases), args.cases)

    model = OpenAILLMModel(
        config=OpenAIBackendConfig(
            base_url=args.base_url or DEFAULT_BASE_URL,
            model=args.model or DEFAULT_MODEL,
            temperature=args.temperature,
            top_p=args.top_p,
            max_new_tokens=args.max_new_tokens,
            reasoning=False,
        )
    )
    generation_config = GenerationConfig(
        prefix_window_tokens=args.prefix_window_tokens,
        consistency_pass_enabled=not args.disable_consistency_pass,
        prompt_language=args.prompt_language,
    )
    pipeline = ChunkWiseGenerationPipeline(
        model=model,
        tokenizer=WhitespaceTokenizer(),
        config=generation_config,
    )

    case_results: list[dict[str, Any]] = []
    for case in cases:
        case_result = _run_case(
            case=case,
            pipeline=pipeline,
            model=model,
            prompt_language=args.prompt_language,
        )
        case_results.append(case_result)

    summary = _aggregate_summary(case_results)
    output_payload = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "cases_path": str(args.cases),
        "config": {
            "model": args.model or DEFAULT_MODEL,
            "base_url": args.base_url or DEFAULT_BASE_URL,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "max_new_tokens": args.max_new_tokens,
            "prefix_window_tokens": args.prefix_window_tokens,
            "prompt_language": args.prompt_language,
            "consistency_pass_enabled": not args.disable_consistency_pass,
        },
        "summary": summary,
        "cases": case_results,
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_json = args.output_dir / "ab_baseline_report.json"
    report_md = args.output_dir / "ab_baseline_report.md"
    report_json.write_text(
        json.dumps(output_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    _write_markdown_report(path=report_md, payload=output_payload)

    for result in case_results:
        case_file = args.output_dir / f"{_sanitize_filename(result['case_id'])}.json"
        case_file.write_text(
            json.dumps(result, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    logger.info("Saved baseline JSON report to %s", report_json)
    logger.info("Saved baseline Markdown report to %s", report_md)
    logger.info(
        "Summary: cases=%d, chunked_wins=%d, one_shot_wins=%d, ties=%d",
        summary["total_cases"],
        summary["chunked_wins"],
        summary["one_shot_wins"],
        summary["ties"],
    )


if __name__ == "__main__":
    main()
