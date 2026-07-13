"""
Parallel Computation — execute feature computation in parallel using
thread or process pool executors.

Wraps the existing ``FeatureComputationEngine`` from ``src.feature_store``
with parallel execution, progress tracking, and error handling.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from typing import Any, Callable

from src.feature_framework.models import ComputationResult

logger = logging.getLogger(__name__)


def make_thread_pool(max_workers: int | None = None) -> ThreadPoolExecutor:
    """Create a ThreadPoolExecutor with sensible defaults.

    Parameters
    ----------
    max_workers : int, optional
        Max worker threads. Defaults to ``min(32, os.cpu_count() + 4)``.

    Returns
    -------
    ThreadPoolExecutor
    """
    import os
    n_workers = max_workers or min(32, (os.cpu_count() or 4) + 4)
    return ThreadPoolExecutor(max_workers=n_workers)


def make_process_pool(max_workers: int | None = None) -> ProcessPoolExecutor:
    """Create a ProcessPoolExecutor with sensible defaults.

    Parameters
    ----------
    max_workers : int, optional
        Max worker processes. Defaults to ``os.cpu_count()``.

    Returns
    -------
    ProcessPoolExecutor
    """
    import os
    n_workers = max_workers or (os.cpu_count() or 4)
    return ProcessPoolExecutor(max_workers=n_workers)


class ParallelComputer:
    """Execute feature computation in parallel across entities.

    Wraps a computation function and distributes entity IDs across
    a thread or process pool executor.

    Parameters
    ----------
    compute_fn : callable
        Function ``(entity_id: int, **kwargs) -> ComputationResult``.
    executor : str
        ``thread`` or ``process`` (default ``thread``).
    max_workers : int, optional
        Max parallel workers.
    show_progress : bool
        Log progress at INFO level (default True).
    """

    def __init__(
        self,
        compute_fn: Callable[..., ComputationResult],
        executor: str = "thread",
        max_workers: int | None = None,
        show_progress: bool = True,
    ) -> None:
        self.compute_fn = compute_fn
        self.executor_type = executor
        self.max_workers = max_workers
        self.show_progress = show_progress

    def run(
        self,
        entity_ids: list[int],
        **kwargs: Any,
    ) -> list[ComputationResult]:
        """Compute features for multiple entities in parallel.

        Parameters
        ----------
        entity_ids : list[int]
            Entity IDs to process.
        **kwargs
            Additional keyword arguments passed to ``compute_fn``.

        Returns
        -------
        list[ComputationResult]
            Results for each entity (order matches input).
        """
        if not entity_ids:
            return []

        # Choose executor
        if self.executor_type == "process":
            pool_cls = ProcessPoolExecutor
        else:
            pool_cls = ThreadPoolExecutor

        import os
        n_workers = self.max_workers or (
            (os.cpu_count() or 4) if self.executor_type == "process"
            else min(32, (os.cpu_count() or 4) + 4)
        )

        results: list[ComputationResult] = []
        start_time = time.time()
        n_total = len(entity_ids)

        with pool_cls(max_workers=n_workers) as executor:
            # Submit all tasks
            future_map = {
                executor.submit(self.compute_fn, eid, **kwargs): eid
                for eid in entity_ids
            }

            # Collect results as they complete
            completed = 0
            for future in as_completed(future_map):
                eid = future_map[future]
                try:
                    result = future.result()
                    results.append(result)
                except Exception as exc:
                    results.append(ComputationResult(
                        feature_name=kwargs.get("feature_name", "unknown"),
                        entity_id=eid,
                        entity_type=kwargs.get("entity_type", "match"),
                        success=False,
                        error=str(exc),
                    ))

                completed += 1
                if self.show_progress and (completed % max(1, n_total // 10) == 0):
                    pct = completed / n_total * 100
                    elapsed = time.time() - start_time
                    logger.info(
                        "  Parallel: %d/%d (%.0f%%) in %.1fs",
                        completed, n_total, pct, elapsed,
                    )

        # Restore original order by entity_id
        # Use len(entity_ids) as sentinel so unknown IDs sort to the end
        id_order = {eid: i for i, eid in enumerate(entity_ids)}
        _sentinel = len(entity_ids)
        results.sort(key=lambda r: id_order.get(r.entity_id, _sentinel))

        elapsed = time.time() - start_time
        n_ok = sum(1 for r in results if r.success)
        n_fail = sum(1 for r in results if not r.success)
        logger.info(
            "Parallel compute: %d/%d ok, %d failed (%.2fs, %d workers)",
            n_ok, n_total, n_fail, elapsed, n_workers,
        )

        return results

    def run_batch(
        self,
        entity_ids: list[int],
        batch_size: int = 100,
        **kwargs: Any,
    ) -> list[ComputationResult]:
        """Compute features in parallel batches.

        Useful for very large entity sets where batching prevents
        memory issues.

        Parameters
        ----------
        entity_ids : list[int]
            Entity IDs to process.
        batch_size : int
            Entities per batch (default 100).
        **kwargs
            Passed to ``run()``.

        Returns
        -------
        list[ComputationResult]
        """
        all_results: list[ComputationResult] = []
        for i in range(0, len(entity_ids), batch_size):
            batch = entity_ids[i:i + batch_size]
            batch_results = self.run(batch, **kwargs)
            all_results.extend(batch_results)
            logger.info(
                "Batch %d/%d: %d entities",
                i // batch_size + 1, (len(entity_ids) + batch_size - 1) // batch_size,
                len(batch),
            )
        return all_results
