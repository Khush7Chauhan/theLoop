class ToolError(Exception):
    def __init__(self, error_type: str, message: str, retryable: bool):
        super().__init__(message)
        self.error_type = error_type
        self.message = message
        self.retryable = retryable
