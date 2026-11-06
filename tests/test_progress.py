from multiprocessing.pool import ThreadPool

from visionsim.utils.progress import PoolProgress


def _advance(update, count):
    for _ in range(count):
        update(advance=1)


def test_waits_for_multiple_tasks_from_a_pool():
    with PoolProgress(description="test") as progress:
        updates = [progress.add_task(f"task-{i}", total=5) for i in range(3)]

        with ThreadPool(len(updates)) as pool:
            result = pool.starmap_async(_advance, [(update, 5) for update in updates])
            progress.wait()
            result.get()

        assert len(progress.completed_tasks) == len(updates)
