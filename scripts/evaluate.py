"""CLI evaluation script — compare predicted MangaPage JSONs against ground truth."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from rich.console import Console
from rich.table import Table

from mangaka.eval.detection import detection_metrics, per_class_metrics
from mangaka.eval.generation import layout_validity_score, page_summary
from mangaka.schema import ELEMENT_TYPES, MangaPage


def _load_pages(directory: Path) -> list[tuple[str, MangaPage]]:
    """Load all .json files from a directory, sorted by name."""
    pages = []
    for path in sorted(directory.glob("*.json")):
        try:
            page = MangaPage.from_json(path)
            pages.append((path.stem, page))
        except Exception as e:
            print(f"Warning: skipping {path.name}: {e}", file=sys.stderr)
    return pages


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate manga detection predictions against ground truth.",
    )
    parser.add_argument(
        "--predictions-dir",
        type=Path,
        required=True,
        help="Directory of predicted MangaPage JSONs.",
    )
    parser.add_argument(
        "--ground-truth-dir",
        type=Path,
        required=True,
        help="Directory of ground truth MangaPage JSONs.",
    )
    parser.add_argument(
        "--iou-thresholds",
        type=str,
        default="0.5,0.75",
        help="Comma-separated IoU thresholds (default: 0.5,0.75).",
    )
    args = parser.parse_args()

    console = Console()
    thresholds = [float(t) for t in args.iou_thresholds.split(",")]

    # Load pages
    pred_items = _load_pages(args.predictions_dir)
    gt_items = _load_pages(args.ground_truth_dir)

    if not pred_items:
        console.print("[red]No prediction files found.[/red]")
        sys.exit(1)
    if not gt_items:
        console.print("[red]No ground truth files found.[/red]")
        sys.exit(1)

    # Align by filename
    gt_map = {name: page for name, page in gt_items}
    matched_pred = []
    matched_gt = []
    for name, page in pred_items:
        if name in gt_map:
            matched_pred.append(page)
            matched_gt.append(gt_map[name])

    if not matched_pred:
        console.print("[red]No matching filenames between prediction and ground truth dirs.[/red]")
        sys.exit(1)

    console.print(f"\n[bold]Evaluating {len(matched_pred)} pages[/bold]\n")

    # Overall detection metrics
    overall = detection_metrics(matched_pred, matched_gt, iou_thresholds=thresholds)

    table = Table(title="Overall Detection Metrics")
    table.add_column("IoU Threshold", style="cyan")
    table.add_column("Precision", style="green")
    table.add_column("Recall", style="green")
    table.add_column("F1", style="bold green")
    table.add_column("TP / Pred / GT", style="dim")

    for thresh_str, m in overall["thresholds"].items():
        prec = f"{m['precision']:.3f}" if not _isnan(m["precision"]) else "N/A"
        table.add_row(
            thresh_str,
            prec,
            f"{m['recall']:.3f}",
            f"{m['f1']:.3f}",
            f"{m['tp']} / {m['num_predictions']} / {m['num_ground_truth']}",
        )

    console.print(table)
    console.print(f"\n[bold]Mean F1:[/bold] {overall['mean_f1']:.3f}\n")

    # Per-class metrics (at first threshold)
    pc = per_class_metrics(matched_pred, matched_gt, iou_threshold=thresholds[0])

    class_table = Table(title=f"Per-Class Metrics (IoU={thresholds[0]})")
    class_table.add_column("Type", style="cyan")
    class_table.add_column("Precision", style="green")
    class_table.add_column("Recall", style="green")
    class_table.add_column("F1", style="bold green")
    class_table.add_column("TP / Pred / GT", style="dim")

    for etype in ELEMENT_TYPES:
        m = pc[etype]
        prec = f"{m['precision']:.3f}" if not _isnan(m["precision"]) else "N/A"
        class_table.add_row(
            etype,
            prec,
            f"{m['recall']:.3f}",
            f"{m['f1']:.3f}",
            f"{m['tp']} / {m['num_predictions']} / {m['num_ground_truth']}",
        )

    console.print(class_table)

    # Layout validity for predictions
    valid_scores = []
    for page in matched_pred:
        v = layout_validity_score(page)
        valid_scores.append(v["score"])

    mean_validity = sum(valid_scores) / len(valid_scores) if valid_scores else 0.0
    console.print(f"\n[bold]Mean Layout Validity:[/bold] {mean_validity:.3f}\n")


def _isnan(x: float) -> bool:
    try:
        return x != x  # NaN != NaN
    except Exception:
        return False


if __name__ == "__main__":
    main()
