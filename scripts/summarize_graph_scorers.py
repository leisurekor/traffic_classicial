"""Thin unified entrypoint for graph scorer-family summaries.

This wrapper exists so callers do not need to remember the narrower
paper-inspired script name. It delegates directly to the existing scorer-family
summary implementation and preserves the same CLI surface.
"""

from __future__ import annotations

from summarize_paper_inspired_scorers import main


if __name__ == "__main__":
    raise SystemExit(main())
