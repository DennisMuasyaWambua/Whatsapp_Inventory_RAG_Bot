#!/usr/bin/env python3

import multiprocessing
import os

# Server socket
bind = "0.0.0.0:8000"
backlog = 2048

# Worker processes
workers = 1  # Single worker to prevent memory competition
worker_class = "sync"
worker_connections = 1000
timeout = 120
keepalive = 2

# Memory management
max_requests = 100  # Restart workers after 100 requests to prevent memory leaks
max_requests_jitter = 10
preload_app = False  # Don't preload to save memory

# Process naming
proc_name = 'whatsapp_inventory_bot'

# User and group to run as
user = None
group = None

# Logging
loglevel = 'info'
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s"'

# Memory limits (Linux only)
limit_memory_hard = 1024 * 1024 * 1024  # 1GB hard limit
limit_memory_soft = 768 * 1024 * 1024   # 768MB soft limit

# Worker timeout
graceful_timeout = 30
worker_tmp_dir = "/dev/shm"  # Use RAM disk for worker temp files

def when_ready(server):
    print("Gunicorn server is ready with memory optimization")
    
def worker_int(worker):
    print(f"Worker {worker.pid} received INT signal, cleaning up...")
    
def worker_abort(worker):
    print(f"Worker {worker.pid} was aborted due to memory limits")

def on_exit(server):
    print("Gunicorn server shutting down")