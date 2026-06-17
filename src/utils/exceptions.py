from __future__ import annotations


def exception_message(exc: Exception) -> str:
    message = str(exc).strip()
    return message if message else type(exc).__name__
