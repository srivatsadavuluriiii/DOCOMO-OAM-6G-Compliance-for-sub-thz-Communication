#!/usr/bin/env python3
"""
Comprehensive exception handling system for OAM 6G.

This module provides robust exception handling with graceful degradation,
proper logging, and user-friendly error messages for all critical operations.
"""

import logging
import traceback
import sys
from typing import Any, Callable, Optional, Type, Union, Dict
from functools import wraps
import numpy as np

                   
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class OAMException(Exception):
    """Base exception class for OAM 6G system."""
    
    def __init__(self, message: str, error_code: str = None, details: Dict[str, Any] = None):
        super().__init__(message)
        self.error_code = error_code
        self.details = details or {}
        self.timestamp = logging.Formatter().formatTime(logging.LogRecord(
            'exception', logging.ERROR, '', 0, '', (), None
        ))

class CalculationException(OAMException):
    """Exception raised when calculations fail."""
    pass

class ConfigurationException(OAMException):
    """Exception raised when configuration is invalid."""
    pass

class FileOperationException(OAMException):
    """Exception raised when file operations fail."""
    pass

class ValidationException(OAMException):
    """Exception raised when validation fails."""
    pass

class DegradationException(OAMException):
    """Exception raised when graceful degradation is needed."""
    pass

class ExceptionHandler:
    """
    Comprehensive exception handler with graceful degradation.
    
    Provides:
    - Graceful degradation for calculation failures
    - Proper logging with context
    - User-friendly error messages
    - Fallback values for critical operations
    """
    
    def __init__(self):
        """Initialize the exception handler."""
        self.fallback_values = {
            'sinr_dB': -10.0,                                       
            'throughput': 0.0,                                             
            'position': np.array([100.0, 0.0, 2.0]),                    
            'velocity': np.array([0.0, 0.0, 0.0]),                    
            'channel_matrix': np.eye(6, dtype=complex),                          
            'reward': -1.0,                                         
            'action': 0,                  
            'mode': 1,                    
        }
        
        self.error_counts = {}
        self.max_errors_per_operation = 10
    
    def handle_calculation_error(self, operation: str, error: Exception, 
                               fallback_value: Any = None, **kwargs) -> Any:
        """
        Handle calculation errors with graceful degradation.
        
        Args:
            operation: Name of the operation that failed
            error: The exception that occurred
            fallback_value: Fallback value to return
            **kwargs: Additional context for logging
            
        Returns:
            Fallback value or default fallback
        """
        error_type = type(error).__name__
        error_key = f"{operation}_{error_type}"
        
                      
        self.error_counts[error_key] = self.error_counts.get(error_key, 0) + 1
        
                                    
        logger.warning(
            f"Calculation error in {operation}: {error}",
            extra={
                'operation': operation,
                'error_type': error_type,
                'error_count': self.error_counts[error_key],
                **kwargs
            }
        )
        
                                          
        if fallback_value is not None:
            return fallback_value
        
                                                       
        if 'sinr' in operation.lower():
            return self.fallback_values['sinr_dB']
        elif 'throughput' in operation.lower():
            return self.fallback_values['throughput']
        elif 'position' in operation.lower():
            return self.fallback_values['position']
        elif 'velocity' in operation.lower():
            return self.fallback_values['velocity']
        elif 'channel' in operation.lower():
            return self.fallback_values['channel_matrix']
        elif 'reward' in operation.lower():
            return self.fallback_values['reward']
        else:
            return None
    
    def handle_file_operation_error(self, operation: str, file_path: str, 
                                  error: Exception, **kwargs) -> None:
        """
        Handle file operation errors.
        
        Args:
            operation: Name of the file operation
            file_path: Path to the file
            error: The exception that occurred
            **kwargs: Additional context for logging
        """
        error_type = type(error).__name__
        
        logger.error(
            f"File operation error in {operation} for {file_path}: {error}",
            extra={
                'operation': operation,
                'file_path': file_path,
                'error_type': error_type,
                **kwargs
            }
        )
        
                                         
        raise FileOperationException(
            f"Failed to {operation} file '{file_path}': {str(error)}",
            error_code="FILE_OP_ERROR",
            details={'operation': operation, 'file_path': file_path}
        )
    
    def handle_validation_error(self, validation_type: str, value: Any, 
                              expected: Any, **kwargs) -> None:
        """
        Handle validation errors.
        
        Args:
            validation_type: Type of validation that failed
            value: The value that failed validation
            expected: Expected value or range
            **kwargs: Additional context for logging
        """
        logger.error(
            f"Validation error in {validation_type}: got {value}, expected {expected}",
            extra={
                'validation_type': validation_type,
                'actual_value': value,
                'expected_value': expected,
                **kwargs
            }
        )
        
        raise ValidationException(
            f"Validation failed for {validation_type}: got {value}, expected {expected}",
            error_code="VALIDATION_ERROR",
            details={'validation_type': validation_type, 'actual': value, 'expected': expected}
        )
    
    def safe_execute(self, operation: str, func: Callable, *args, 
                    fallback_value: Any = None, max_retries: int = 1, **kwargs) -> Any:
        """
        Safely execute a function with error handling and retries.
        
        Args:
            operation: Name of the operation
            func: Function to execute
            *args: Arguments for the function
            fallback_value: Fallback value if execution fails
            max_retries: Maximum number of retries
            **kwargs: Additional context for logging
            
        Returns:
            Function result or fallback value
        """
        for attempt in range(max_retries + 1):
            try:
                return func(*args)
            except Exception as e:
                if attempt == max_retries:
                    logger.error(
                        f"Operation {operation} failed after {max_retries + 1} attempts: {e}",
                        extra={'operation': operation, 'attempts': max_retries + 1, **kwargs}
                    )
                    return self.handle_calculation_error(operation, e, fallback_value, **kwargs)
                else:
                    logger.warning(
                        f"Operation {operation} failed (attempt {attempt + 1}/{max_retries + 1}): {e}",
                        extra={'operation': operation, 'attempt': attempt + 1, **kwargs}
                    )
        
        return fallback_value

def safe_calculation(operation_name: str, fallback_value: Any = None, 
                    max_retries: int = 1, log_level: str = "warning"):
    """
    Decorator for safe calculation with error handling.
    
    Args:
        operation_name: Name of the operation for logging
        fallback_value: Fallback value if calculation fails
        max_retries: Maximum number of retries
        log_level: Logging level for errors
        
    Returns:
        Decorated function with error handling
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = ExceptionHandler()
            
            for attempt in range(max_retries + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries:
                        if log_level == "error":
                            logger.error(f"Calculation failed in {operation_name}: {e}")
                        else:
                            logger.warning(f"Calculation failed in {operation_name}: {e}")
                        return handler.handle_calculation_error(operation_name, e, fallback_value)
                    else:
                        logger.debug(f"Retry {attempt + 1}/{max_retries} for {operation_name}")
            
            return fallback_value
        return wrapper
    return decorator

def safe_file_operation(operation_name: str):
    """
    Decorator for safe file operations with error handling.
    
    Args:
        operation_name: Name of the file operation
        
    Returns:
        Decorated function with error handling
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = ExceptionHandler()
            
            try:
                return func(*args, **kwargs)
            except Exception as e:
                                                         
                file_path = "unknown"
                if args and isinstance(args[0], str):
                    file_path = args[0]
                elif 'file_path' in kwargs:
                    file_path = kwargs['file_path']
                
                handler.handle_file_operation_error(operation_name, file_path, e)
            
            return None
        return wrapper
    return decorator

def validate_input(validation_func: Callable, error_message: str = None):
    """
    Decorator for input validation with error handling.
    
    Args:
        validation_func: Function that performs validation
        error_message: Custom error message
        
    Returns:
        Decorated function with validation
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            handler = ExceptionHandler()
            
            try:
                                    
                validation_result = validation_func(*args, **kwargs)
                if not validation_result:
                    raise ValidationException(
                        error_message or f"Validation failed for {func.__name__}"
                    )
                return func(*args, **kwargs)
            except Exception as e:
                handler.handle_validation_error(
                    f"{func.__name__}_validation",
                    args,
                    "valid input",
                    error=str(e)
                )
            
            return None
        return wrapper
    return decorator

def graceful_degradation(default_value: Any = None, log_warning: bool = True):
    """
    Decorator for graceful degradation when operations fail.
    
    Args:
        default_value: Default value to return on failure
        log_warning: Whether to log warnings
        
    Returns:
        Decorated function with graceful degradation
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                if log_warning:
                    logger.warning(
                        f"Operation {func.__name__} failed, using default value: {e}"
                    )
                return default_value
        return wrapper
    return decorator

                                   
exception_handler = ExceptionHandler()

def get_exception_handler() -> ExceptionHandler:
    """Get the global exception handler instance."""
    return exception_handler 
