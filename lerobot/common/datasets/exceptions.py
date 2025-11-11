class MissingAnnotatedTasksError(FileNotFoundError):
    """Raised when annotated tasks are requested but the annotations file is missing."""
