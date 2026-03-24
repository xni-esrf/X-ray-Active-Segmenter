from .config import AppConfig

__all__ = [
    "AppConfig",
    "AppContext",
    "run",
]


def __getattr__(name: str):
    if name in {"AppContext", "run"}:
        from .app import AppContext, run

        globals()["AppContext"] = AppContext
        globals()["run"] = run
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
