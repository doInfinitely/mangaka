"""
CLI: Build the vector index over a manga JSON dataset.

Usage:
    python scripts/index_dataset.py \
        --dataset-dir data/manga_dataset \
        --index-dir data/manga_index
"""

from __future__ import annotations

import argparse
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("mangaka.index_dataset")


def main():
    parser = argparse.ArgumentParser(description="Build manga RAG index")
    parser.add_argument("--dataset-dir", required=True,
                        help="Path to the manga JSON dataset (output of build_dataset)")
    parser.add_argument("--index-dir", default="data/manga_index",
                        help="Where to save the index")
    parser.add_argument("--embed-model", default="all-MiniLM-L6-v2",
                        help="Sentence-transformers model for embeddings")
    parser.add_argument("--batch-size", type=int, default=64)
    args = parser.parse_args()

    from mangaka.rag.index import MangaIndex

    index = MangaIndex()
    index.build(
        dataset_dir=args.dataset_dir,
        embed_model=args.embed_model,
        batch_size=args.batch_size,
    )
    index.save(args.index_dir)
    logger.info("Index saved to %s", args.index_dir)


if __name__ == "__main__":
    main()
