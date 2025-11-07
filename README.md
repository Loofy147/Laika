# AI Memory System

This project is a proof-of-concept for a 'Memory and Identity AI Stack'. Its purpose is to create AI with a dynamic, updatable memory based on user interactions to enable long-term, personalized experiences.

## Core Concepts

*   **Event-Driven Memory:** The AI's memory is updated only when significant events are detected in the interaction history.
*   **Dynamic Identity:** The user's identity is represented by a deep embedding that can be updated over time.
*   **Continuous Learning:** The AI's memory update function is continuously trained on new events.
*   **Differential Equation-Based Memory:** The memory state evolves according to a differential equation, which allows for a more natural decay of memories over time.

## Architecture

The system is composed of the following modules:

*   `core.py`: The main module that integrates all the other components.
*   `memory_controller.py`: Manages the AI's memory state and the neural network that computes memory updates.
*   `identity_module.py`: Represents the user's identity and handles property encoding.
*   `adaptive_event_detector.py`: Detects significant events in the interaction history with an adaptive threshold.
*   `api.py`: A Flask-based REST API for interacting with the AI system.

## Getting Started

### Prerequisites

*   Python 3.9+
*   Docker (optional)

### Installation

1.  Clone the repository:
    ```bash
    git clone https://github.com/your-username/ai-memory-system.git
    cd ai-memory-system
    ```

2.  Run the build script:
    ```bash
    python3 build.py
    ```
    This will create a virtual environment, install the dependencies, and run the tests.

### Running the Simulation

To run the simulation, execute the following command:
```bash
python3 -m ai_memory_system.main
```

### Running the API

To run the API, execute the following command:
```bash
python3 -m ai_memory_system.api
```

### Running with Docker

1.  Build the Docker image:
    ```bash
    docker build -t ai-memory-system .
    ```

2.  Run the Docker container:
    ```bash
    docker run -p 5000:5000 ai-memory-system
    ```

## Usage

### API Endpoints

*   `GET /memory`: Returns the current memory state.
*   `POST /interact`: Processes a new interaction.

Example `POST` request to `/interact`:
```json
{
    "type": "chat",
    "content": "What is the meaning of life?",
    "significance": 0.8
}
```
