# Configuration for the AI Memory System

# --- Model Hyperparameters ---
MEMORY_SIZE = 128
"""The size of the memory state vector."""

IDENTITY_EMBEDDING_SIZE = 384
"""The size of the identity embedding."""

EVENT_EMBEDDING_SIZE = 384
"""The size of the event embedding."""

HIDDEN_SIZE = 512
"""The hidden size for the Transformer model."""

NHEAD = 4
"""The number of attention heads in the Transformer model."""

NUM_LAYERS = 2
"""The number of layers in the Transformer model."""


# --- Training Hyperparameters ---
LEARNING_RATE = 0.001
"""The initial learning rate for the AdamW optimizer."""

PATIENCE = 5
"""
The number of epochs to wait for improvement before reducing the learning
rate.
"""

MIN_DELTA = 1e-5
"""
The minimum change in the monitored quantity to qualify as an improvement.
"""

MAX_GRAD_NORM = 1.0
"""The maximum norm of the gradients for gradient clipping."""


# --- Memory Controller Hyperparameters ---
LAMBDA_DECAY = 0.01
"""The decay rate for the memory state."""

ACTIVATION_FACTOR = 1.0
"""The activation factor for the memory update."""


# --- Event Detector Hyperparameters ---
EVENT_THRESHOLD = 0.6
"""The initial threshold for detecting significant events."""

WINDOW_SIZE = 10
"""
The size of the sliding window for calculating the running average of
significance scores.
"""

ADJUSTMENT_FACTOR = 0.1
"""
The factor by which the event threshold is adjusted based on the running
average.
"""


# --- Identity Module Hyperparameters ---
EPSILON = 1.0
"""
The epsilon value for the Laplace mechanism in differential privacy. A
smaller value means more privacy.
"""

ALPHA = 0.1
"""
The alpha value for the exponential moving average when updating the identity
embedding.
"""
