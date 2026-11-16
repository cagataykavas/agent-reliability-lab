from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from .io import load_cases
from .models import EvaluationCase

DEFAULT_MAX_COMPARISONS = 1_000_000
DEFAULT_MAX_FINDINGS = 100
_TOKEN_PATTERN = re.compile(r"[^\W_]+(?:[-'][^\W_]+)*", re.UNICODE)


class OverlapAuditError(ValueError):
    """Raised when a benchmark split cannot be audited safely."""


@dataclass(frozen=True, slots=True)
class OverlapFinding:
    reference_case_id: str
    candidate_case_id: str
    reason: str
    similarity: float
    shared_token_count: int


@dataclass(frozen=True, slots=True)
class OverlapReport:
    admitted: bool
    reference_cases: int
    candidate_cases: int
    comparisons: int
    similarity_threshold: float
    max_allowed_overlaps: int
    overlap_count: int
    truncated_findings: int
    findings: tuple[OverlapFinding, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def normalize_prompt(prompt: str) -> str:
    normalized = unicodedata.normalize("NFKC", prompt).casefold()
    return " ".join(_TOKEN_PATTERN.findall(normalized))


def prompt_tokens(prompt: str) -> frozenset[str]:
    return frozenset(normalize_prompt(prompt).split())


def token_jaccard(left: frozenset[str], right: frozenset[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 1.0


def _validate_split(name: str, cases: Sequence[EvaluationCase]) -> None:
    if not cases:
        raise OverlapAuditError(f"{name} split must contain at least one case")
    seen: set[str] = set()
    for case in cases:
        if not case.case_id.strip():
            raise OverlapAuditError(f"{name} split contains an empty case_id")
        if case.case_id in seen:
            raise OverlapAuditError(f"{name} split contains duplicate case_id: {case.case_id}")
        seen.add(case.case_id)
        if not normalize_prompt(case.prompt):
            raise OverlapAuditError(f"{name} case {case.case_id} has an empty prompt")


def audit_overlap(
    reference: Sequence[EvaluationCase],
    candidate: Sequence[EvaluationCase],
    *,
    similarity_threshold: float = 0.8,
    max_overlaps: int = 0,
    max_comparisons: int = DEFAULT_MAX_COMPARISONS,
    max_findings: int = DEFAULT_MAX_FINDINGS,
) -> OverlapReport:
    if not 0.5 <= similarity_threshold <= 1.0:
        raise OverlapAuditError("similarity_threshold must be between 0.5 and 1.0")
    if max_overlaps < 0:
        raise OverlapAuditError("max_overlaps cannot be negative")
    if max_comparisons < 1:
        raise OverlapAuditError("max_comparisons must be positive")
    if max_findings < 1:
        raise OverlapAuditError("max_findings must be positive")
    _validate_split("reference", reference)
    _validate_split("candidate", candidate)

    comparison_count = len(reference) * len(candidate)
    if comparison_count > max_comparisons:
        raise OverlapAuditError(
            f"audit requires {comparison_count} comparisons, exceeding the "
            f"configured limit of {max_comparisons}"
        )

    reference_features = [
        (case, normalize_prompt(case.prompt), prompt_tokens(case.prompt)) for case in reference
    ]
    candidate_features = [
        (case, normalize_prompt(case.prompt), prompt_tokens(case.prompt)) for case in candidate
    ]

    overlap_count = 0
    findings: list[OverlapFinding] = []
    for reference_case, reference_prompt, reference_tokens in reference_features:
        for candidate_case, candidate_prompt, candidate_tokens in candidate_features:
            shared_tokens = len(reference_tokens & candidate_tokens)
            similarity = token_jaccard(reference_tokens, candidate_tokens)
            reason: str | None = None
            if reference_case.case_id == candidate_case.case_id:
                reason = "case_id_collision"
            elif reference_prompt == candidate_prompt:
                reason = "exact_prompt_match"
            elif similarity >= similarity_threshold:
                reason = "near_duplicate_prompt"

            if reason is None:
                continue
            overlap_count += 1
            if len(findings) < max_findings:
                findings.append(
                    OverlapFinding(
                        reference_case_id=reference_case.case_id,
                        candidate_case_id=candidate_case.case_id,
                        reason=reason,
                        similarity=round(similarity, 6),
                        shared_token_count=shared_tokens,
                    )
                )

    findings.sort(
        key=lambda finding: (
            -finding.similarity,
            finding.reference_case_id,
            finding.candidate_case_id,
        )
    )
    return OverlapReport(
        admitted=overlap_count <= max_overlaps,
        reference_cases=len(reference),
        candidate_cases=len(candidate),
        comparisons=comparison_count,
        similarity_threshold=similarity_threshold,
        max_allowed_overlaps=max_overlaps,
        overlap_count=overlap_count,
        truncated_findings=max(0, overlap_count - len(findings)),
        findings=tuple(findings),
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Fail closed when candidate agent cases overlap a reference benchmark split."
    )
    parser.add_argument("reference_cases", help="trusted/reference JSONL evaluation cases")
    parser.add_argument("candidate_cases", help="candidate/held-out JSONL evaluation cases")
    parser.add_argument("--similarity-threshold", type=float, default=0.8)
    parser.add_argument("--max-overlaps", type=int, default=0)
    parser.add_argument("--max-comparisons", type=int, default=DEFAULT_MAX_COMPARISONS)
    parser.add_argument("--max-findings", type=int, default=DEFAULT_MAX_FINDINGS)
    args = parser.parse_args(argv)

    try:
        report = audit_overlap(
            load_cases(args.reference_cases),
            load_cases(args.candidate_cases),
            similarity_threshold=args.similarity_threshold,
            max_overlaps=args.max_overlaps,
            max_comparisons=args.max_comparisons,
            max_findings=args.max_findings,
        )
    except (OSError, ValueError) as error:
        print(json.dumps({"error": str(error)}, sort_keys=True), file=sys.stderr)
        return 1

    print(json.dumps(report.to_dict(), indent=2, sort_keys=True))
    return 0 if report.admitted else 2


if __name__ == "__main__":
    raise SystemExit(main())
