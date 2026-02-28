"""Server entry point — launch the mangaka REST API with uvicorn."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Launch the Mangaka REST API server.")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Bind host (default: 0.0.0.0).")
    parser.add_argument("--port", type=int, default=8000, help="Bind port (default: 8000).")
    parser.add_argument(
        "--model-dir",
        type=Path,
        default=Path("checkpoints"),
        help="Directory with model checkpoints (default: checkpoints/).",
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("data/manga_dataset"),
        help="Directory with dataset (default: data/manga_dataset/).",
    )
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload for development.",
    )
    args = parser.parse_args()

    # Configure app state before importing (to avoid circular imports at module level)
    from mangaka.server.app import state

    state.model_dir = args.model_dir
    state.dataset_dir = args.dataset_dir

    import uvicorn

    uvicorn.run(
        "mangaka.server.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
