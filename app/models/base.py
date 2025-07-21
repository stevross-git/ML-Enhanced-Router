"""
Base Model Class
Provides common functionality for all database models
"""

from datetime import datetime
from sqlalchemy.orm import Mapped, mapped_column, DeclarativeBase
from sqlalchemy import DateTime, Integer
from sqlalchemy.ext.declarative import declared_attr
import uuid

class Base(DeclarativeBase):
    """Base class for all database models"""
    
    @declared_attr
    def __tablename__(cls):
        """Auto-generate table name from class name"""
        # Convert CamelCase to snake_case
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', cls.__name__)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

class TimestampMixin:
    """Mixin for adding timestamp fields to models"""
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=datetime.utcnow,
        nullable=False
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime,
        default=datetime.utcnow,
        onupdate=datetime.utcnow,
        nullable=False
    )

class SoftDeleteMixin:
    """Mixin for soft delete functionality"""
    
    deleted_at: Mapped[datetime | None] = mapped_column(
        DateTime,
        nullable=True,
        default=None
    )
    
    @property
    def is_deleted(self) -> bool:
        """Check if record is soft deleted"""
        return self.deleted_at is not None
    
    def soft_delete(self):
        """Mark record as soft deleted"""
        self.deleted_at = datetime.utcnow()
    
    def restore(self):
        """Restore soft deleted record"""
        self.deleted_at = None

class UUIDMixin:
    """Mixin for UUID primary keys"""
    
    @declared_attr
    def id(cls):
        return mapped_column(
            'id',
            String(36),
            primary_key=True,
            default=lambda: str(uuid.uuid4())
        )

def generate_id():
    """Generate a unique ID for database records"""
    return str(uuid.uuid4())

# Export commonly used classes
__all__ = [
    'Base',
    'TimestampMixin', 
    'SoftDeleteMixin',
    'UUIDMixin',
    'generate_id'
]
