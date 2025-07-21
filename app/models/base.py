"""
Base Model Classes
Common base classes and mixins for database models
"""

import uuid
from datetime import datetime, timezone
from sqlalchemy import String, DateTime, Boolean, Integer
from sqlalchemy.ext.declarative import declared_attr
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

# Create base class for all models
class Base(DeclarativeBase):
    """Base class for all database models"""
    
    @declared_attr
    def __tablename__(cls):
        """Auto-generate table name from class name"""
        # Convert CamelCase to snake_case
        import re
        name = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', cls.__name__)
        return re.sub('([a-z0-9])([A-Z])', r'\1_\2', name).lower()

def generate_id():
    """Generate a new UUID4 ID"""
    return str(uuid.uuid4())

class UUIDMixin:
    """Mixin for models that use UUID as primary key"""
    
    @declared_attr
    def id(cls):
        return mapped_column(
            String(36),
            primary_key=True,
            default=generate_id
        )

class TimestampMixin:
    """Mixin for models that need created/updated timestamps"""
    
    created_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=lambda: datetime.now(timezone.utc), 
        nullable=False
    )
    
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, 
        default=lambda: datetime.now(timezone.utc), 
        onupdate=lambda: datetime.now(timezone.utc), 
        nullable=False
    )

class SoftDeleteMixin:
    """Mixin for models that support soft deletion"""
    
    is_deleted: Mapped[bool] = mapped_column(Boolean, default=False, nullable=False)
    deleted_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    
    def soft_delete(self):
        """Mark record as deleted"""
        self.is_deleted = True
        self.deleted_at = datetime.now(timezone.utc)
    
    def restore(self):
        """Restore soft-deleted record"""
        self.is_deleted = False
        self.deleted_at = None

# Export commonly used classes
__all__ = [
    'Base',
    'TimestampMixin', 
    'SoftDeleteMixin',
    'UUIDMixin',
    'generate_id'
]