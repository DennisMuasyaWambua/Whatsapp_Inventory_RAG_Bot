import psutil
import gc
import os
import logging
from functools import wraps

def monitor_memory(func):
    """Decorator to monitor memory usage of functions"""
    @wraps(func)
    def wrapper(*args, **kwargs):
        process = psutil.Process(os.getpid())
        
        # Get initial memory usage
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        logging.info(f"Starting {func.__name__} - Memory: {initial_memory:.2f} MB")
        
        try:
            result = func(*args, **kwargs)
            
            # Get final memory usage
            final_memory = process.memory_info().rss / 1024 / 1024  # MB
            memory_diff = final_memory - initial_memory
            
            logging.info(f"Completed {func.__name__} - Memory: {final_memory:.2f} MB (+{memory_diff:.2f} MB)")
            
            # Force cleanup if memory usage increased significantly
            if memory_diff > 100:  # More than 100MB increase
                logging.warning(f"High memory increase detected ({memory_diff:.2f} MB), forcing cleanup")
                force_cleanup()
            
            return result
            
        except Exception as e:
            logging.error(f"Error in {func.__name__}: {e}")
            force_cleanup()
            raise
    
    return wrapper

def force_cleanup():
    """Force aggressive memory cleanup"""
    try:
        # Python garbage collection
        collected = gc.collect()
        
        # Clear torch cache if available
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except ImportError:
            pass
        
        # Linux memory trim
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except:
            pass
        
        process = psutil.Process(os.getpid())
        current_memory = process.memory_info().rss / 1024 / 1024
        
        logging.info(f"Memory cleanup completed - Collected {collected} objects - Memory: {current_memory:.2f} MB")
        
    except Exception as e:
        logging.error(f"Error during memory cleanup: {e}")

def check_memory_limit(limit_mb=1024):
    """Check if memory usage exceeds limit and cleanup if needed"""
    process = psutil.Process(os.getpid())
    current_memory = process.memory_info().rss / 1024 / 1024
    
    if current_memory > limit_mb:
        logging.warning(f"Memory limit exceeded: {current_memory:.2f} MB > {limit_mb} MB")
        force_cleanup()
        
        # Check again after cleanup
        new_memory = process.memory_info().rss / 1024 / 1024
        if new_memory > limit_mb:
            logging.critical(f"Memory still high after cleanup: {new_memory:.2f} MB")
            raise MemoryError(f"Memory usage {new_memory:.2f} MB exceeds limit {limit_mb} MB")
    
    return current_memory

class MemoryLimitedProcessor:
    """Context manager for memory-limited processing"""
    
    def __init__(self, memory_limit_mb=768, cleanup_threshold_mb=512):
        self.memory_limit = memory_limit_mb
        self.cleanup_threshold = cleanup_threshold_mb
        self.process = psutil.Process(os.getpid())
        self.initial_memory = None
        
    def __enter__(self):
        self.initial_memory = self.process.memory_info().rss / 1024 / 1024
        logging.info(f"Starting memory-limited processing - Initial: {self.initial_memory:.2f} MB")
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        final_memory = self.process.memory_info().rss / 1024 / 1024
        memory_diff = final_memory - self.initial_memory
        
        logging.info(f"Memory-limited processing completed - Final: {final_memory:.2f} MB (+{memory_diff:.2f} MB)")
        
        # Always cleanup on exit
        force_cleanup()
        
    def check_and_cleanup(self):
        """Check memory usage and cleanup if needed"""
        current_memory = self.process.memory_info().rss / 1024 / 1024
        
        if current_memory > self.cleanup_threshold:
            logging.info(f"Memory threshold reached ({current_memory:.2f} MB), cleaning up")
            force_cleanup()
            
        if current_memory > self.memory_limit:
            raise MemoryError(f"Memory limit exceeded: {current_memory:.2f} MB > {self.memory_limit} MB")