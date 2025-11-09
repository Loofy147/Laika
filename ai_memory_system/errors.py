class AppError(Exception):
    """Base class for application errors."""
    def __init__(self, message, status_code=500, error_code=None):
        super().__init__(message)
        self.status_code = status_code
        self.error_code = error_code

class AgentNotFoundError(AppError):
    """Raised when an agent is not found."""
    def __init__(self, user_id):
        super().__init__(f"Agent for user '{user_id}' not found.", status_code=404, error_code="AGENT_NOT_FOUND")

class ValidationError(AppError):
    """Raised when input validation fails."""
    def __init__(self, message):
        super().__init__(message, status_code=400, error_code="VALIDATION_ERROR")
