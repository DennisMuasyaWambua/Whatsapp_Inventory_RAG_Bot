"""
Memory monitoring and limiting utility for the WhatsApp Inventory RAG Bot.
This prevents system crashes by monitoring and limiting memory usage.
"""

import psutil
import gc
import os
import logging
from functools import wraps

logger = logging.getLogger(__name__)

class MemoryLimiter:
    """Monitor and limit memory usage to prevent system crashes."""
    
    def __init__(self, max_memory_mb=1000, warning_threshold_mb=800):
        """
        Initialize memory limiter.
        
        Args:
            max_memory_mb: Maximum memory in MB before forcing cleanup
            warning_threshold_mb: Memory threshold for warnings
        """
        self.max_memory_mb = max_memory_mb
        self.warning_threshold_mb = warning_threshold_mb
        self.process = psutil.Process(os.getpid())
        self.initial_memory = self.get_current_memory()
        
    def get_current_memory(self):
        """Get current memory usage in MB."""
        return self.process.memory_info().rss / 1024 / 1024
    
    def get_memory_usage(self):
        """Get memory usage relative to initial."""
        current = self.get_current_memory()
        return current - self.initial_memory
    
    def check_memory_limit(self, raise_on_limit=True):
        """
        Check if memory usage exceeds limits.
        
        Args:
            raise_on_limit: Whether to raise MemoryError on limit exceeded
            
        Returns:
            bool: True if within limits, False otherwise
        """
        current_usage = self.get_memory_usage()
        current_total = self.get_current_memory()
        
        if current_usage > self.max_memory_mb:
            logger.error(f"Memory limit exceeded: {current_total:.2f} MB (used {current_usage:.2f} MB)")
            if raise_on_limit:
                raise MemoryError(f"Memory limit of {self.max_memory_mb} MB exceeded")
            return False
            
        if current_usage > self.warning_threshold_mb:
            logger.warning(f"Memory warning: {current_total:.2f} MB (used {current_usage:.2f} MB)")
            
        return True
    
    def force_cleanup(self):
        """Force garbage collection and memory cleanup."""
        logger.info("Forcing memory cleanup...")
        
        # Force garbage collection multiple times
        for _ in range(3):
            gc.collect()
        
        # Clear CUDA cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        # Try to trim memory (Linux only)
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except:
            pass
        
        logger.info(f"Memory after cleanup: {self.get_current_memory():.2f} MB")

def memory_monitor(max_memory_mb=1000):
    """
    Decorator to monitor memory usage of functions.
    
    Args:
        max_memory_mb: Maximum memory usage allowed in MB
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            limiter = MemoryLimiter(max_memory_mb=max_memory_mb)
            
            try:
                # Check memory before execution
                limiter.check_memory_limit()
                
                # Execute function
                result = func(*args, **kwargs)
                
                # Check memory after execution
                limiter.check_memory_limit()
                
                return result
                
            except MemoryError:
                # Attempt cleanup before re-raising
                limiter.force_cleanup()
                raise
                
        return wrapper
    return decorator

def check_system_resources():
    """
    Check if system has enough resources to continue processing.
    
    Returns:
        tuple: (bool, str) - (can_continue, message)
    """
    # Check available memory
    memory = psutil.virtual_memory()
    available_gb = memory.available / 1024 / 1024 / 1024
    
    if available_gb < 1.0:
        return False, f"Low system memory: {available_gb:.1f}GB available"
    
    # Check CPU usage
    cpu_percent = psutil.cpu_percent(interval=1)
    if cpu_percent > 90:
        return False, f"High CPU usage: {cpu_percent}%"
    
    # Check disk space
    disk = psutil.disk_usage('/')
    free_gb = disk.free / 1024 / 1024 / 1024
    
    if free_gb < 1.0:
        return False, f"Low disk space: {free_gb:.1f}GB available"
    
    return True, "System resources OK"