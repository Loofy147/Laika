"""
Thread-safe user state management.

Pattern: Per-Resource Lock
References: Java ConcurrentHashMap, Python threading
"""

import threading
from contextlib import contextmanager
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class UserStateLockManager:
    """
    Thread-safe user state management with fine-grained locking.

    Features:
    - Per-user locks (not global) for maximum concurrency
    - Timeout to prevent deadlocks
    - Context manager for automatic cleanup
    """

    def __init__(self, timeout=5.0):
        self._locks = defaultdict(threading.RLock)
        self._lock_for_locks = threading.Lock()
        self.timeout = timeout
        logger.info(f"LockManager initialized: timeout={timeout}s")

    @contextmanager
    def user_lock(self, user_id: str):
        """
        Context manager for user-specific lock.

        Usage:
            with lock_manager.user_lock('user_123'):
                # Thread-safe operations
                pass
        """
        lock = self._locks[user_id]
        acquired = lock.acquire(timeout=self.timeout)

        if not acquired:
            logger.error(f"Lock timeout for user: {user_id}")
            raise TimeoutError(f"Lock timeout for {user_id}")

        try:
            yield
        finally:
            lock.release()
