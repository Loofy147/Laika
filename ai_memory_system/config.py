# Configuration for the AI Memory System

# Model Hyperparameters
MEMORY_SIZE = 128
IDENTITY_EMBEDDING_SIZE = 384
EVENT_EMBEDDING_SIZE = 384
HIDDEN_SIZE = 512
NHEAD = 4
NUM_LAYERS = 2

# Training Hyperparameters
LEARNING_RATE = 0.001
PATIENCE = 5
MIN_DELTA = 1e-5
MAX_GRAD_NORM = 1.0

# Memory Controller Hyperparameters
LAMBDA_DECAY = 0.01
ACTIVATION_FACTOR = 1.0

# Event Detector Hyperparameters
EVENT_THRESHOLD = 0.6
WINDOW_SIZE = 10
ADJUSTMENT_FACTOR = 0.1

# Replay Buffer
REPLAY_BUFFER_CAPACITY = 1000
BATCH_SIZE = 32

# Identity Module Hyperparameters
EPSILON = 1.0
ALPHA = 0.1
