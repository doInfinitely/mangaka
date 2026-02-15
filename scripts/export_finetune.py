"""
CLI: Export manga dataset as LLM finetuning data.

Usage:
    # Chat format (OpenAI / Anthropic finetuning)
    python scripts/export_finetune.py \
        --dataset-dir data/manga_dataset \
        --output data/finetune/train_chat.jsonl \
        --format chat

    # Completion format
    python scripts/export_finetune.py \
        --dataset-dir data/manga_dataset \
        --output data/finetune/train_completion.jsonl \
        --format completion

    # Instruction format (Alpaca-style)
    python scripts/export_finetune.py \
        --dataset-dir data/manga_dataset \
        --output data/finetune/train_instruction.jsonl \
        --format instruction

    # Multi-page sequence format
    python scripts/export_finetune.py \
        --dataset-dir data/manga_dataset \
        --output data/finetune/train_sequence.jsonl \
        --format sequence \
        --pages-per-sequence 4
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("mangaka.export_finetune")


def main():
    parser = argparse.ArgumentParser(description="Export finetuning data")
    parser.add_argument("--dataset-dir", required=True,
                        help="Path to manga JSON dataset")
    parser.add_argument("--output", required=True,
                        help="Output JSONL file path")
    parser.add_argument("--format", choices=["chat", "completion", "instruction", "sequence"],
                        default="chat")
    parser.add_argument("--max-pages", type=int, default=None,
                        help="Limit number of pages to export")
    parser.add_argument("--pages-per-sequence", type=int, default=4,
                        help="Pages per sequence (for sequence format)")
    parser.add_argument("--max-sequences", type=int, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    from mangaka.rag.finetune_data import (
        export_chat_jsonl,
        export_completion_jsonl,
        export_instruction_jsonl,
        export_sequence_jsonl,
    )

    if args.format == "chat":
        count = export_chat_jsonl(
            args.dataset_dir, args.output, args.max_pages, args.seed,
        )
    elif args.format == "completion":
        count = export_completion_jsonl(
            args.dataset_dir, args.output, args.max_pages, args.seed,
        )
    elif args.format == "instruction":
        count = export_instruction_jsonl(
            args.dataset_dir, args.output, args.max_pages, args.seed,
        )
    elif args.format == "sequence":
        count = export_sequence_jsonl(
            args.dataset_dir, args.output,
            args.pages_per_sequence, args.max_sequences, args.seed,
        )

    logger.info("Exported %d examples to %s", count, args.output)


if __name__ == "__main__":
    main()
