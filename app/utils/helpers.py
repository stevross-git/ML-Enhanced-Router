"""
Helper Utilities
Common utility functions used across the application
"""

import hashlib
import json
import uuid
import time
import re
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Union
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

def generate_session_id() -> str:
    """Generate a unique session ID"""
    return str(uuid.uuid4())

def generate_cache_key(query: str, model_id: str, parameters: Dict = None) -> str:
    """Generate cache key for query"""
    # Create a string representation of all inputs
    cache_data = {
        'query': query,
        'model_id': model_id,
        'parameters': parameters or {}
    }
    
    # Convert to JSON string for consistent hashing
    cache_string = json.dumps(cache_data, sort_keys=True)
    
    # Generate hash
    return hashlib.md5(cache_string.encode()).hexdigest()

def hash_query(query: str) -> str:
    """Generate hash for query text"""
    return hashlib.sha256(query.encode()).hexdigest()

def format_response_time(response_time: float) -> str:
    """Format response time for display"""
    if response_time < 1:
        return f"{response_time * 1000:.0f}ms"
    elif response_time < 60:
        return f"{response_time:.1f}s"
    else:
        minutes = int(response_time // 60)
        seconds = response_time % 60
        return f"{minutes}m {seconds:.1f}s"

def format_file_size(size_bytes: int) -> str:
    """Format file size in human-readable format"""
    if size_bytes == 0:
        return "0B"
    
    size_names = ["B", "KB", "MB", "GB", "TB"]
    i = 0
    while size_bytes >= 1024 and i < len(size_names) - 1:
        size_bytes /= 1024.0
        i += 1
    
    return f"{size_bytes:.1f}{size_names[i]}"

def format_number(number: Union[int, float]) -> str:
    """Format large numbers with appropriate suffixes"""
    if number >= 1000000:
        return f"{number / 1000000:.1f}M"
    elif number >= 1000:
        return f"{number / 1000:.1f}K"
    else:
        return str(number)

def calculate_cost(tokens: int, model_id: str = None) -> float:
    """Calculate approximate cost based on tokens"""
    # Rough cost estimates per 1K tokens (input + output average)
    cost_per_1k = {
        'gpt-4': 0.035,
        'gpt-4-turbo': 0.015,
        'gpt-3.5-turbo': 0.002,
        'claude-3-opus': 0.0225,
        'claude-3-sonnet': 0.006,
        'claude-3-haiku': 0.00075,
        'gemini-pro': 0.00025,
        'default': 0.01
    }
    
    rate = cost_per_1k.get(model_id, cost_per_1k['default'])
    return (tokens / 1000) * rate

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text"""
    # Simple keyword extraction
    # Remove punctuation and convert to lowercase
    text = re.sub(r'[^\w\s]', '', text.lower())
    
    # Split into words
    words = text.split()
    
    # Remove common stop words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'from', 'up', 'about', 'into', 'through', 'during',
        'before', 'after', 'above', 'below', 'between', 'among', 'under', 'over',
        'is', 'are', 'was', 'were', 'be', 'been', 'being', 'have', 'has', 'had',
        'do', 'does', 'did', 'will', 'would', 'could', 'should', 'may', 'might',
        'must', 'can', 'shall', 'this', 'that', 'these', 'those', 'i', 'you',
        'he', 'she', 'it', 'we', 'they', 'me', 'him', 'her', 'us', 'them'
    }
    
    # Filter words
    keywords = [word for word in words if word not in stop_words and len(word) > 2]
    
    # Count frequency
    word_freq = {}
    for word in keywords:
        word_freq[word] = word_freq.get(word, 0) + 1
    
    # Sort by frequency and return top keywords
    sorted_keywords = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
    return [word for word, freq in sorted_keywords[:max_keywords]]

def truncate_text(text: str, max_length: int = 100, suffix: str = "...") -> str:
    """Truncate text to specified length"""
    if len(text) <= max_length:
        return text
    
    return text[:max_length - len(suffix)] + suffix

def clean_filename(filename: str) -> str:
    """Clean filename for safe storage"""
    # Remove or replace problematic characters
    filename = re.sub(r'[<>:"/\\|?*]', '_', filename)
    
    # Remove leading/trailing spaces and dots
    filename = filename.strip(' .')
    
    # Ensure it's not empty
    if not filename:
        filename = f"file_{int(time.time())}"
    
    return filename

def merge_dictionaries(*dicts: Dict[str, Any]) -> Dict[str, Any]:
    """Merge multiple dictionaries"""
    result = {}
    for d in dicts:
        if d:
            result.update(d)
    return result

def deep_get(dictionary: Dict[str, Any], key_path: str, default: Any = None) -> Any:
    """Get nested dictionary value using dot notation"""
    keys = key_path.split('.')
    value = dictionary
    
    try:
        for key in keys:
            value = value[key]
        return value
    except (KeyError, TypeError):
        return default

def deep_set(dictionary: Dict[str, Any], key_path: str, value: Any) -> None:
    """Set nested dictionary value using dot notation"""
    keys = key_path.split('.')
    current = dictionary
    
    for key in keys[:-1]:
        if key not in current or not isinstance(current[key], dict):
            current[key] = {}
        current = current[key]
    
    current[keys[-1]] = value

def is_valid_json(text: str) -> bool:
    """Check if string is valid JSON"""
    try:
        json.loads(text)
        return True
    except (json.JSONDecodeError, TypeError):
        return False

def safe_json_loads(text: str, default: Any = None) -> Any:
    """Safely load JSON with default fallback"""
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default

def safe_json_dumps(obj: Any, default: Any = None) -> str:
    """Safely dump JSON with error handling"""
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except (TypeError, ValueError):
        return json.dumps(default or {})

def retry_operation(func, max_retries: int = 3, delay: float = 1.0, 
                   backoff_factor: float = 2.0, exceptions: tuple = (Exception,)):
    """Retry operation with exponential backoff"""
    def wrapper(*args, **kwargs):
        last_exception = None
        current_delay = delay
        
        for attempt in range(max_retries + 1):
            try:
                return func(*args, **kwargs)
            except exceptions as e:
                last_exception = e
                if attempt < max_retries:
                    logger.warning(f"Attempt {attempt + 1} failed: {e}. Retrying in {current_delay}s...")
                    time.sleep(current_delay)
                    current_delay *= backoff_factor
                else:
                    logger.error(f"All {max_retries + 1} attempts failed. Last error: {e}")
        
        raise last_exception
    
    return wrapper

def create_directory(path: Union[str, Path], exist_ok: bool = True) -> Path:
    """Create directory with error handling"""
    path_obj = Path(path)
    try:
        path_obj.mkdir(parents=True, exist_ok=exist_ok)
        return path_obj
    except Exception as e:
        logger.error(f"Failed to create directory {path}: {e}")
        raise

def get_file_extension(filename: str) -> str:
    """Get file extension"""
    return Path(filename).suffix.lower().lstrip('.')

def is_expired(timestamp: datetime, ttl_seconds: int) -> bool:
    """Check if timestamp has expired"""
    if not timestamp:
        return True
    
    expiry_time = timestamp + timedelta(seconds=ttl_seconds)
    return datetime.utcnow() > expiry_time

def timestamp_to_string(timestamp: datetime, format_str: str = "%Y-%m-%d %H:%M:%S") -> str:
    """Convert timestamp to formatted string"""
    if not timestamp:
        return ""
    
    return timestamp.strftime(format_str)

def string_to_timestamp(date_string: str, format_str: str = "%Y-%m-%d %H:%M:%S") -> Optional[datetime]:
    """Convert string to timestamp"""
    try:
        return datetime.strptime(date_string, format_str)
    except ValueError:
        return None

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate simple text similarity"""
    # Simple Jaccard similarity
    set1 = set(text1.lower().split())
    set2 = set(text2.lower().split())
    
    intersection = len(set1.intersection(set2))
    union = len(set1.union(set2))
    
    if union == 0:
        return 0.0
    
    return intersection / union

def chunk_list(lst: List[Any], chunk_size: int) -> List[List[Any]]:
    """Split list into chunks"""
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def flatten_dict(d: Dict[str, Any], parent_key: str = '', sep: str = '.') -> Dict[str, Any]:
    """Flatten nested dictionary"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)

def get_client_ip(request) -> str:
    """Get client IP address from request"""
    # Check for forwarded headers first
    forwarded_ips = request.headers.get('X-Forwarded-For')
    if forwarded_ips:
        return forwarded_ips.split(',')[0].strip()
    
    real_ip = request.headers.get('X-Real-IP')
    if real_ip:
        return real_ip
    
    return request.remote_addr or 'unknown'

def parse_user_agent(user_agent_string: str) -> Dict[str, str]:
    """Parse user agent string"""
    # Simple user agent parsing
    result = {
        'browser': 'unknown',
        'version': 'unknown',
        'os': 'unknown'
    }
    
    if not user_agent_string:
        return result
    
    # Browser detection
    if 'Chrome' in user_agent_string:
        result['browser'] = 'Chrome'
    elif 'Firefox' in user_agent_string:
        result['browser'] = 'Firefox'
    elif 'Safari' in user_agent_string:
        result['browser'] = 'Safari'
    elif 'Edge' in user_agent_string:
        result['browser'] = 'Edge'
    
    # OS detection
    if 'Windows' in user_agent_string:
        result['os'] = 'Windows'
    elif 'macOS' in user_agent_string or 'Mac OS' in user_agent_string:
        result['os'] = 'macOS'
    elif 'Linux' in user_agent_string:
        result['os'] = 'Linux'
    elif 'Android' in user_agent_string:
        result['os'] = 'Android'
    elif 'iOS' in user_agent_string:
        result['os'] = 'iOS'
    
    return result