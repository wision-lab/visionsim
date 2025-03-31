from __future__ import annotations

from typing_extensions import Protocol


class UpdateFn(Protocol):
    """Mirrors `rich.progress.Progress.update` with curried task-id argument"""

    def __call__(
        self,
        total: float | None = None,
        completed: float | None = None,
        advance: float | None = None,
        description: str | None = None,
        visible: bool | None = None,
        refresh: bool = False,
    ) -> None: ...
