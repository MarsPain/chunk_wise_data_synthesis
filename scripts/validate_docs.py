#!/usr/bin/env python3
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FILES = [
    "AGENTS.md",
    "docs/DESIGN.md",
    "docs/FRONTEND.md",
    "docs/PLANS.md",
    "docs/PRODUCT_SENSE.md",
    "docs/ROADMAP.md",
    "docs/exec-plans/README.md",
    "docs/plans/README.md",
]

REQUIRED_DIRS = [
    "docs/design-docs",
    "docs/exec-plans/active",
    "docs/exec-plans/completed",
    "docs/exec-plans/tech-debt",
    "docs/generated",
    "docs/product-specs",
    "docs/references",
]

REQUIRED_AGENTS_LINKS = [
    "docs/DESIGN.md",
    "docs/FRONTEND.md",
    "docs/PLANS.md",
    "docs/PRODUCT_SENSE.md",
    "docs/ROADMAP.md",
]

LEGACY_REDIRECT_RULES: dict[str, list[str]] = {
    "docs/plans/README.md": [
        "MIGRATION_REDIRECT",
        "../PLANS.md",
        "../exec-plans/README.md",
        "../design-docs/README.md",
    ],
    "docs/plans/2026-02-14-chunk-wise-autoregressive-design.md": [
        "MIGRATION_REDIRECT",
        "../design-docs/chunk-wise-autoregressive-design.md",
    ],
}

MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")


def _strip_fenced_code(markdown: str) -> str:
    kept_lines: list[str] = []
    in_code_fence = False
    for line in markdown.splitlines():
        if line.lstrip().startswith("```"):
            in_code_fence = not in_code_fence
            continue
        if not in_code_fence:
            kept_lines.append(line)
    return "\n".join(kept_lines)


def _normalize_link_target(raw_target: str) -> str:
    target = raw_target.strip()
    if not target:
        return ""

    if target.startswith("<") and target.endswith(">"):
        target = target[1:-1].strip()

    # Handle links with optional titles: (path "title")
    if " " in target and not target.startswith(("http://", "https://")):
        target = target.split(" ", 1)[0]

    target = target.split("#", 1)[0]
    target = target.split("?", 1)[0]
    return target.strip()


def _check_required_paths(errors: list[str]) -> None:
    for relative_path in REQUIRED_FILES:
        path = ROOT / relative_path
        if not path.is_file():
            errors.append(f"Missing required file: {relative_path}")

    for relative_path in REQUIRED_DIRS:
        path = ROOT / relative_path
        if not path.is_dir():
            errors.append(f"Missing required directory: {relative_path}")


def _check_agents_constraints(errors: list[str]) -> None:
    agents_path = ROOT / "AGENTS.md"
    if not agents_path.is_file():
        return

    content = agents_path.read_text(encoding="utf-8")
    line_count = len(content.splitlines())
    if line_count > 140:
        errors.append(
            f"AGENTS.md too long: {line_count} lines (max allowed: 140)"
        )

    for link_fragment in REQUIRED_AGENTS_LINKS:
        if link_fragment not in content:
            errors.append(
                "AGENTS.md missing required core docs link fragment: "
                f"{link_fragment}"
            )


def _check_redirect_docs(errors: list[str]) -> None:
    for relative_path, required_markers in LEGACY_REDIRECT_RULES.items():
        path = ROOT / relative_path
        if not path.is_file():
            errors.append(f"Missing legacy redirect file: {relative_path}")
            continue
        content = path.read_text(encoding="utf-8")
        for marker in required_markers:
            if marker not in content:
                errors.append(
                    f"Redirect file {relative_path} missing marker: {marker}"
                )


def _resolve_internal_link(source_file: Path, target: str) -> Path:
    return (source_file.parent / target).resolve()


def _is_external_link(target: str) -> bool:
    return target.startswith(
        (
            "http://",
            "https://",
            "mailto:",
            "tel:",
            "data:",
        )
    )


def _iter_markdown_files() -> list[Path]:
    files = [ROOT / "AGENTS.md"]
    files.extend(sorted((ROOT / "docs").rglob("*.md")))
    return [f for f in files if f.is_file()]


def _check_markdown_links(errors: list[str]) -> None:
    for md_file in _iter_markdown_files():
        content = md_file.read_text(encoding="utf-8")
        scan_content = _strip_fenced_code(content)
        for match in MARKDOWN_LINK_RE.finditer(scan_content):
            raw_target = match.group(1)
            target = _normalize_link_target(raw_target)
            if not target:
                continue
            if target.startswith("#"):
                continue
            if _is_external_link(target):
                continue
            if target.startswith("/"):
                # Absolute paths are intentionally ignored.
                continue

            resolved = _resolve_internal_link(md_file, target)
            if not resolved.exists():
                rel_source = md_file.relative_to(ROOT)
                errors.append(
                    "Broken internal markdown link: "
                    f"{rel_source} -> {target}"
                )


def _check_plan_hygiene(errors: list[str]) -> None:
    plans_index = ROOT / "docs/PLANS.md"
    index_text = plans_index.read_text(encoding="utf-8") if plans_index.is_file() else ""

    completed_dir = ROOT / "docs/exec-plans/completed"
    tech_debt_dir = ROOT / "docs/exec-plans/tech-debt"

    completed_files = [
        p
        for p in sorted(completed_dir.glob("*.md"))
        if p.name.lower() != "readme.md"
    ]
    if not completed_files:
        errors.append(
            "Plan hygiene violation: docs/exec-plans/completed must contain "
            "at least one completed plan markdown file"
        )

    for plan_file in completed_files:
        rel_from_docs = f"exec-plans/completed/{plan_file.name}"
        if rel_from_docs not in index_text:
            errors.append(
                "PLANS index missing completed plan link fragment: "
                f"{rel_from_docs}"
            )

    tech_debt_files = [
        p
        for p in sorted(tech_debt_dir.glob("*.md"))
        if p.name.lower() != "readme.md"
    ]
    for debt_file in tech_debt_files:
        rel_from_docs = f"exec-plans/tech-debt/{debt_file.name}"
        if rel_from_docs not in index_text:
            errors.append(
                "PLANS index missing tech-debt plan link fragment: "
                f"{rel_from_docs}"
            )

    active_dir = ROOT / "docs/exec-plans/active"
    active_files = [
        p
        for p in sorted(active_dir.glob("*.md"))
        if p.name.lower() != "readme.md"
    ]
    for active_file in active_files:
        rel_from_docs = f"exec-plans/active/{active_file.name}"
        if rel_from_docs not in index_text:
            errors.append(
                "PLANS index missing active plan link fragment: "
                f"{rel_from_docs}"
            )


def validate_docs() -> list[str]:
    errors: list[str] = []
    _check_required_paths(errors)
    _check_agents_constraints(errors)
    _check_redirect_docs(errors)
    _check_markdown_links(errors)
    _check_plan_hygiene(errors)
    return errors


def main() -> int:
    errors = validate_docs()
    if errors:
        print("[validate_docs] FAILED")
        for idx, error in enumerate(errors, start=1):
            print(f"{idx}. {error}")
        return 1

    print("[validate_docs] OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
