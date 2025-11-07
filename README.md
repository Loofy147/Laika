# AI Memory System

This project is a proof-of-concept for a 'Memory and Identity AI Stack'. Its purpose is to create AI with a dynamic, updatable memory based on user interactions to enable long-term, personalized experiences.

## Core Concepts

*   **Event-Driven Memory:** The AI's memory is updated only when significant events are detected in the interaction history.
*   **Dynamic Identity:** The user's identity is represented by a deep embedding that can be updated over time.
*   **Continuous Learning:** The AI's memory update function is continuously trained on new events.
*   **Differential Equation-Based Memory:** The memory state evolves according to a differential equation, which allows for a more natural decay of memories over time.

## Theoretical Analysis

The memory update rule is a discretized version of the following ordinary differential equation:

dM/dt = -λM + a * f_θ(M, I, E)

where:
- M is the memory state
- λ is the decay rate
- a is the activation factor
- f_θ is the neural network that computes the memory update
- I is the user's identity embedding
- E is the event embedding

The stability of this system can be analyzed by examining the eigenvalues of the Jacobian matrix of the system. The system is stable if all eigenvalues have negative real parts. The convergence of the learning process is ensured by the use of the AdamW optimizer and a learning rate scheduler, which adapts the learning rate based on the training loss.

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

*   `POST /login`: Returns a token for a given user.
*   `GET /memory`: Returns the current memory state.
*   `POST /interact`: Processes a new interaction.
*   `GET /explain`: Explains the last memory update.
*   `POST /identity`: Updates the user's properties.

### Authentication

The API uses token-based authentication. To get a token, send a `POST` request to the `/login` endpoint with a JSON body containing your username:

```json
{
    "username": "user1"
}
```

The API will return a token, which you should include in the `Authorization` header of all subsequent requests.

### Updating User Properties

To update the user's properties, send a `POST` request to the `/identity` endpoint with a JSON body containing the new properties:

```json
{
    "interests": ["python", "api_design", "machine_learning"],
    "biography": "I am a software engineer with a passion for AI."
}
```

Example `POST` request to `/interact`:
```json
{
    "type": "chat",
    "content": "What is the meaning of life?",
    "significance": 0.8
}
```

## Performance Evaluation

To evaluate the performance of the AI system, run the `evaluate.py` script:

```bash
python3 evaluate.py
```

This script will compute the following metrics:

*   **Memory Fidelity:** The mean absolute difference between the norms of the target and predicted memory updates.
*   **Learning Stability:** The standard deviation of the training loss.

## Asynchronous Event Handling

In a production environment, it is recommended to handle events asynchronously. This can be achieved by using a message queue like RabbitMQ or Kafka. The API would publish events to the message queue, and a separate worker process would consume the events and update the AI's memory. This architecture would improve the scalability and reliability of the system.

## Extension Points

This project can be extended in several ways:

*   **Advanced Neural Architectures:** The `FTheta` network has been replaced with a Transformer-based model to capture more complex relationships between the memory state, identity, and event.
*   **Multi-Agent Systems:** The system could be extended to support multiple AI agents, each with its own memory and identity. This would allow for the creation of more complex and interactive AI systems.
