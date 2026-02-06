from __future__ import annotations

# Deprecated compatibility entrypoint. Use app.main instead.
from app.main import build_arg_parser, main

__all__ = ["build_arg_parser", "main"]


if __name__ == "__main__":
    raise SystemExit(main())
