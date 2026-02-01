# -*- coding: utf-8 -*-
"""File-based locking utilities for cross-process synchronization.

This module provides locking primitives that work across multiple processes
(e.g., pytest-xdist workers), unlike threading.Lock() which only works within
a single process.

The locking is based on atomic file creation using open(..., 'x') mode, which
is atomic across all platforms (POSIX and Windows).
"""
import contextlib
import os
import time
from pathlib import Path


@contextlib.contextmanager
def acquire_file_lock(lock_file, timeout=60):
    """
    Acquire and release a file-based lock for cross-process synchronization.

    Uses atomic file creation to serialize access to shared resources across
    multiple processes (e.g., pytest-xdist workers). The lock is held for the
    duration of the context manager.

    Args:
        lock_file: Path object or string for the lock file to create/remove
        timeout: Maximum seconds to wait for lock acquisition (default: 60)

    Yields:
        None - lock is acquired when entering context, released on exit

    Raises:
        RuntimeError: If timeout exceeded while waiting for lock
    """
    lock_file = Path(lock_file)
    start_time = time.time()

    # Acquire lock via atomic file creation
    while True:
        try:
            with open(lock_file, 'x') as f:
                f.write(str(os.getpid()))
            break
        except FileExistsError:
            if time.time() - start_time > timeout:
                raise RuntimeError(
                    f"Timeout ({timeout}s) waiting for lock {lock_file}"
                )
            time.sleep(0.01)

    try:
        yield
    finally:
        # Release lock
        try:
            lock_file.unlink()
        except FileNotFoundError:
            pass


@contextlib.contextmanager
def shelve_open(filename, *args, **kwds):
    """
    Context manager for shelve database access with file-based locking.

    Prevents concurrent access from multiple processes (e.g., pytest-xdist workers)
    from corrupting the SQLite WAL database. Uses atomic file creation for
    cross-process synchronization.

    Args:
        filename: Path to shelve database file
        *args, **kwds: Passed to shelve.open()

    Yields:
        shelve.Shelf: Open database object with exclusive access
    """
    import shelve

    filename = str(filename)
    dirname = os.path.dirname(filename)
    if not os.path.exists(dirname):
        os.makedirs(dirname)

    # Create lock file path: store alongside the shelve database
    lock_file = Path(filename).parent / f".{Path(filename).name}.lock"

    # Acquire lock before opening database
    with acquire_file_lock(lock_file):
        sh = shelve.open(filename, "c", *args, **kwds)
        try:
            yield sh
        finally:
            sh.close()