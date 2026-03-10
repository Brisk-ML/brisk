"""MissingService is the value classes assign to self.services when initiated.
""" 

class MissingServices:
    """Placeholder for services before get_services() is called.

    This acts as a sentinel that provides a clear error message when services
    are called without being set first. It also serves as dependency injection
    point for testing.

    Raises
    ------
    RuntimeError
        When a service property is accessed before set_services() is called.
    """
    def __getattr__(self, name: str):
        """Raise error when any service is accessed.

        Parameters
        ----------
        name : str
            Name of the service attribute being accessed
            
        Raises
        ------
        RuntimeError
            When a service property is accessed before set_services() is called.
        """
        raise RuntimeError(
            f"Cannot access service '{name}': Services not configured. "
            f"Call set_services() before using this object. "
            f"This typically happens when the object is created but not "
            f"properly initialized with service dependencies."
        )
    
    def __repr__(self) -> str:
        return "<MissingServices: call set_services() to configure>"
