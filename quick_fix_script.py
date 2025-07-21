#!/usr/bin/env python3
"""
Quick fix script to update agent service metadata references
Run this to automatically fix the metadata field references in agent_service.py
"""

import os
import re
import sys

def fix_agent_service():
    """Fix agent service metadata references"""
    agent_service_path = "app/services/agent_service.py"
    
    if not os.path.exists(agent_service_path):
        print(f"❌ Could not find {agent_service_path}")
        return False
    
    try:
        # Read the file
        with open(agent_service_path, 'r') as f:
            content = f.read()
        
        # Make the replacements
        # Fix 1: register_agent method
        content = re.sub(
            r'metadata=agent_data\.get\(\'metadata\', \{\}\),',
            'agent_metadata=agent_data.get(\'metadata\', {}),',
            content
        )
        
        # Fix 2: update_agent_status method - more complex replacement
        old_pattern = r'if metadata:\s+agent\.metadata\.update\(metadata\)'
        new_replacement = '''if metadata:
            if agent.agent_metadata is None:
                agent.agent_metadata = {}
            agent.agent_metadata.update(metadata)'''
        
        content = re.sub(old_pattern, new_replacement, content, flags=re.MULTILINE)
        
        # Write the file back
        with open(agent_service_path, 'w') as f:
            f.write(content)
        
        print(f"✅ Fixed {agent_service_path}")
        return True
        
    except Exception as e:
        print(f"❌ Error fixing {agent_service_path}: {e}")
        return False

def fix_models_init():
    """Fix models __init__.py import"""
    models_init_path = "app/models/__init__.py"
    
    if not os.path.exists(models_init_path):
        print(f"⚠️  Could not find {models_init_path}")
        return True  # Not critical
    
    try:
        # Read the file
        with open(models_init_path, 'r') as f:
            content = f.read()
        
        # Check if it needs the full import list
        if 'Agent,' not in content and 'AgentCapability' not in content:
            new_content = '''"""
Models Package
Import all models to ensure they are registered with SQLAlchemy
"""

# Import base classes first
from .base import Base, TimestampMixin

# Import all model classes
from .user import User
from .agent import (
    Agent, 
    AgentCapability, 
    AgentSession, 
    AgentMetrics, 
    AgentRegistration  # Legacy model
)

# Export all models for easy importing
__all__ = [
    'Base',
    'TimestampMixin',
    'User',
    'Agent',
    'AgentCapability', 
    'AgentSession',
    'AgentMetrics',
    'AgentRegistration'
]
'''
            
            # Write the file back
            with open(models_init_path, 'w') as f:
                f.write(new_content)
            
            print(f"✅ Fixed {models_init_path}")
        else:
            print(f"ℹ️  {models_init_path} already looks good")
        
        return True
        
    except Exception as e:
        print(f"❌ Error fixing {models_init_path}: {e}")
        return False

def main():
    """Main fix function"""
    print("🔧 Applying quick fixes for agent service...")
    
    success = True
    
    if not fix_agent_service():
        success = False
    
    if not fix_models_init():
        success = False
    
    if success:
        print("🎉 All fixes applied successfully!")
        print("\nNow you can try running:")
        print("python main.py")
    else:
        print("❌ Some fixes failed. Please check the errors above.")
    
    return success

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
