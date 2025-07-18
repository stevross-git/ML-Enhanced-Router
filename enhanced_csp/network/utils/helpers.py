"""Network utility functions."""
import logging
import re
import socket
import time
from typing import Optional, Union

def get_logger(name: str, level: int = logging.INFO) -> logging.Logger:
    """Get a logger instance with proper formatting."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        handler = logging.StreamHandler()
        formatter = logging.Formatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.setLevel(level)
    return logger

def validate_ip_address(ip: str) -> bool:
    """Validate IP address format."""
    try:
        socket.inet_aton(ip)
        return True
    except socket.error:
        try:
            socket.inet_pton(socket.AF_INET6, ip)
            return True
        except socket.error:
            return False

def validate_port_number(port: Union[int, str]) -> int:
    """Validate and return port number."""
    try:
        port_int = int(port)
        if not (1 <= port_int <= 65535):
            raise ValueError(f"Port {port_int} is out of valid range (1-65535)")
        return port_int
    except (ValueError, TypeError):
        raise ValueError(f"Invalid port number: {port}")

def format_bytes(bytes_val: int) -> str:
    """Format bytes in human readable format."""
    if bytes_val < 0:
        return "0 B"
    
    units = ['B', 'KB', 'MB', 'GB', 'TB', 'PB']
    size = float(bytes_val)
    
    for unit in units:
        if size < 1024.0:
            return f"{size:.1f} {unit}"
        size /= 1024.0
    
    return f"{size:.1f} {units[-1]}"

def format_duration(seconds: float) -> str:
    """Format duration in human readable format."""
    if seconds < 0:
        return "0s"
    
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = seconds / 60
        return f"{minutes:.1f}m"
    elif seconds < 86400:
        hours = seconds / 3600
        return f"{hours:.1f}h"
    else:
        days = seconds / 86400
        return f"{days:.1f}d"

def get_local_ip() -> str:
    """Get local IP address."""
    try:
        # Connect to a remote address to get local IP
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        return local_ip
    except Exception:
        return "127.0.0.1"

def is_port_available(port: int, host: str = "127.0.0.1") -> bool:
    """Check if a port is available."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            s.bind((host, port))
            return True
    except OSError:
        return False
