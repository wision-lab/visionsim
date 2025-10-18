from __future__ import annotations

import multiprocessing
from functools import partial

from rich.progress import (
    BarColumn,
    Progress,
    ProgressColumn,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeRemainingColumn,
)

from visionsim.types import UpdateFn


class ElapsedProgress(Progress):
    @classmethod
    def get_default_columns(cls) -> tuple[ProgressColumn, ...]:
        """Overrides `rich.progress.Progress`'s default columns to enable showing elapsed time when finished."""
        return (
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(elapsed_when_finished=True),
        )


class PoolProgress(Progress):
    """Convenience wrapper around rich's `Progress` to enable progress bars when
    using multiple processes. All progressbar updates are carried out by the main
    process, and worker processes communicate their state via a callback obtained
    when a task gets added.

    Example:
        .. code-block:: python


            import multiprocessing

            def long_task(tick, min_len=50, max_len=200):
                import random, time

                length = random.randint(min_len, max_len)
                tick(total=length)

                for _ in range(length):
                    time.sleep(0.01)
                    tick(advance=1)


            if __name__ == "__main__":
                with multiprocessing.Pool(4) as pool, PoolProgress() as progress:
                    for i in range(25):
                        tick = progress.add_task(f"Task: {i}")
                        pool.apply_async(long_task, (tick, ))
                    progress.wait()
                    pool.close()
                    pool.join()
    """

    def __init__(self, *args, auto_visible=True, description="[green]Total progress:", **kwargs) -> None:
        """Initialize a `PoolProgress` instance.

        Note:
            All other *args and **kwargs are passed as is to `rich.progress.Progress`.

        Args:
            auto_visible (bool, optional): if true, automatically hides tasks that have not started
                or finished tasks. Defaults to True.
            description (str, optional): text description for the overall progress.
                Defaults to "[green]Total progress:".
        """
        self.manager: multiprocessing.managers.SyncManager | None = None
        self.progress_queue: multiprocessing.Queue | None = None
        self.overall_taskid: TaskID | None = None
        self.inflight_tasks: set[TaskID] = set()
        self.completed_tasks: set[TaskID] = set()
        self.auto_visible = auto_visible
        self.description = description
        super().__init__(*args, **kwargs)

    @classmethod
    def get_default_columns(cls) -> tuple[ProgressColumn, ...]:
        """Overrides `rich.progress.Progress`'s default columns to enable showing elapsed time when finished."""
        return (
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            TimeRemainingColumn(elapsed_when_finished=True),
        )

    @staticmethod
    def update_task(progress: multiprocessing.Queue[dict], task_id: TaskID, **kwargs) -> None:
        progress.put(kwargs | dict(task_id=task_id))

    def _update_task(self, task_update: dict):
        """Actually perform the queued task update"""
        if self.auto_visible:
            task_update |= dict(visible=True)
        self.update(**task_update)

    def task_percentage(self, task_id: TaskID) -> float:
        with self._lock:
            return self._tasks[task_id].percentage

    def task_finished(self, task_id: TaskID) -> bool:
        with self._lock:
            return self._tasks[task_id].finished

    def add_task(self, *args, **kwargs) -> UpdateFn:  # type: ignore[override]
        """Same as `Progress.add_task` except it returns a callback to update the task
        instead of the task-id. The returned callback is roughly equivalent to `Progress.update`
        with it's first argument (the task-id) already filled out, except calling it will not
        immediately update the task's status. The main process will perform the update asynchronously.
        """
        if self.progress_queue is None:
            raise RuntimeError("Cannot add task if context manager has not been entered.")
        if self.auto_visible:
            kwargs["visible"] = False
        task_id = super().add_task(*args, **kwargs)
        update = partial(self.update_task, self.progress_queue, task_id)
        self.inflight_tasks.add(task_id)
        return update

    def __enter__(self):
        self.start()
        self.manager = multiprocessing.Manager().__enter__()
        self.overall_taskid = super().add_task(self.description)
        self.progress_queue = self.manager.Queue()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.manager.__exit__(exc_type, exc_val, exc_tb)
        return super().__exit__(exc_type, exc_val, exc_tb)

    def wait(self) -> None:
        """Block and wait for tasks to finish.

        Note:
            This is what actually updates the progress bars, if not called before exiting the
            with-block no progress will be reported, and processes might be killed.
        """
        if self.progress_queue is None or self.overall_taskid is None:
            raise RuntimeError("Cannot wait on tasks outside of context manager.")

        while self.inflight_tasks:
            while not self.progress_queue.empty():
                task_update = self.progress_queue.get()
                task_id = task_update.get("task_id")
                self._update_task(task_update)

                if self.task_finished(task_id):
                    if self.auto_visible:
                        self.update(task_id, visible=False)
                    if task_id in self.inflight_tasks:
                        self.completed_tasks.add(task_id)
                        self.inflight_tasks.remove(task_id)

                task_progress = sum(self.task_percentage(t) / 100 for t in self.inflight_tasks)
                self.update(
                    self.overall_taskid,
                    completed=len(self.completed_tasks) + task_progress,
                    total=len(self.completed_tasks) + len(self.inflight_tasks),
                )
        self.update(
            self.overall_taskid,
            completed=len(self.completed_tasks),
            total=len(self.completed_tasks),
        )
