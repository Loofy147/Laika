import os

# Configuration for the AI Memory System

# Model Hyperparameters
MEMORY_SIZE = int(os.environ.get("MEMORY_SIZE", 128))
IDENTITY_EMBEDDING_SIZE = int(os.environ.get("IDENTITY_EMBEDDING_SIZE", 384))
EVENT_EMBEDDING_SIZE = int(os.environ.get("EVENT_EMBEDDING_SIZE", 384))
HIDDEN_SIZE = int(os.environ.get("HIDDEN_SIZE", 512))
NHEAD = int(os.environ.get("NHEAD", 4))
NUM_LAYERS = int(os.environ.get("NUM_LAYERS", 2))

# Training Hyperparameters
LEARNING_RATE = float(os.environ.get("LEARNING_RATE", 0.001))
PATIENCE = int(os.environ.get("PATIENCE", 5))
MIN_DELTA = float(os.environ.get("MIN_DELTA", 1e-5))
MAX_GRAD_NORM = float(os.environ.get("MAX_GRAD_NORM", 1.0))

# Memory Controller Hyperparameters
LAMBDA_DECAY = float(os.environ.get("LAMBDA_DECAY", 0.01))
ACTIVATION_FACTOR = float(os.environ.get("ACTIVATION_FACTOR", 1.0))

# Event Detector Hyperparameters
EVENT_THRESHOLD = float(os.environ.get("EVENT_THRESHOLD", 0.6))
WINDOW_SIZE = int(os.environ.get("WINDOW_SIZE", 10))
ADJUSTMENT_FACTOR = float(os.environ.get("ADJUSTMENT_FACTOR", 0.1))

# Replay Buffer
REPLAY_BUFFER_CAPACITY = int(os.environ.get("REPLAY_BUFFER_CAPACITY", 1000))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", 32))

# Identity Module Hyperparameters
EPSILON = float(os.environ.get("EPSILON", 1.0))
ALPHA = float(os.environ.get("ALPHA", 0.1))

# File Paths
DATA_DIR = os.environ.get("DATA_DIR", "agent_data")
