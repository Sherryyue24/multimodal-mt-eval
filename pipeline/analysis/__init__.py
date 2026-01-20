# analysis module - evaluation and aggregation
from .scoring import CometKiwiScorer, score_predictions
from .judging import GPTJudge, run_pairwise_judging
from .aggregation import (
    aggregate_scores_by_mode,
    aggregate_scores_by_language,
    aggregate_judge_results,
    generate_summary_report
)
from .summary import (
    generate_cometkiwi_summary,
    generate_judge_summary,
    generate_text_report,
    generate_full_summary
)
from .case_selection import select_case_studies, export_case_studies
from .error_taxonomy import detect_issues, run_error_analysis

__all__ = [
    # Scoring
    "CometKiwiScorer",
    "score_predictions",
    # Judging
    "GPTJudge",
    "run_pairwise_judging",
    # Aggregation (legacy)
    "aggregate_scores_by_mode",
    "aggregate_scores_by_language",
    "aggregate_judge_results",
    "generate_summary_report",
    # Summary (new)
    "generate_cometkiwi_summary",
    "generate_judge_summary",
    "generate_text_report",
    "generate_full_summary",
    # Case studies
    "select_case_studies",
    "export_case_studies",
    # Error analysis
    "detect_issues",
    "run_error_analysis",
]