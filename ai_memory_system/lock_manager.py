import threading

class LockManager:
    """A thread-safe manager for user-specific locks."""
    def __init__(self):
        self._locks = {}
        self._master_lock = threading.Lock()

    def get_lock(self, user_id):
        """Get the lock for a specific user, creating it if it doesn't exist."""
        with self._master_lock:
            if user_id not in self._locks:
                self._locks[user_id] = threading.Lock()
            return self._locks[user_id]
