#!/usr/bin/env python
"""Launcher: python agent_mvp/evaluation/fedeval.py <validate|models|pull|run|score> ..."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from fed_eval.cli import main  # noqa: E402

raise SystemExit(main())
