"""
Centralized configuration with environment variable support.

Follows 12-factor app methodology.
"""

import os


def get_env_int(key, default):
    """Get integer from environment."""
    return int(os.environ.get(key, default))


def get_env_float(key, default):
    """Get float from environment."""
    return float(os.environ.get(key, default))


# Model Hyperparameters
MEMORY_SIZE = get_env_int('MEMORY_SIZE', 128)
IDENTITY_EMBEDDING_SIZE = get_env_int('IDENTITY_EMBEDDING_SIZE', 384)
EVENT_EMBEDDING_SIZE = get_env_int('EVENT_EMBEDDING_SIZE', 384)
HIDDEN_SIZE = get_env_int('HIDDEN_SIZE', 512)
NHEAD = get_env_int('NHEAD', 4)
NUM_LAYERS = get_env_int('NUM_LAYERS', 2)


# Training
LEARNING_RATE = get_env_float('LEARNING_RATE', 0.001)
PATIENCE = get_env_int('PATIENCE', 5)
MIN_DELTA = get_env_float('MIN_DELTA', 1e-5)
MAX_GRAD_NORM = get_env_float('MAX_GRAD_NORM', 1.0)

# Replay Buffer
REPLAY_BUFFER_SIZE = get_env_int('REPLAY_BUFFER_SIZE', 10000)
REPLAY_SAMPLE_RATIO = get_env_float('REPLAY_SAMPLE_RATIO', 0.5)

# Memory Dynamics
LAMBDA_DECAY = get_env_float('LAMBDA_DECAY', 0.01)
ACTIVATION_FACTOR = get_env_float('ACTIVATION_FACTOR', 1.0)

# Event Detector Hyperparameters
EVENT_THRESHOLD = get_env_float('EVENT_THRESHOLD', 0.6)
WINDOW_SIZE = get_env_int('WINDOW_SIZE', 10)
ADJUSTMENT_FACTOR = get_env_float('ADJUSTMENT_FACTOR', 0.1)


# Privacy
EPSILON = get_env_float('PRIVACY_EPSILON', 5.0)
ALPHA = get_env_float('ALPHA', 0.1)

# API
USER_LOCK_TIMEOUT = get_env_float('USER_LOCK_TIMEOUT', 5.0)
MAX_REQUEST_SIZE = get_env_int('MAX_REQUEST_SIZE', 5000)
