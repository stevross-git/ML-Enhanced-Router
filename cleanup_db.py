#!/usr/bin/env python3
"""
Database cleanup script
"""
from app import app, db
from app.models.model import MLModelRegistry

def cleanup_invalid_models():
    """Clean up models with invalid types"""
    with app.app_context():
        try:
            # Get all models
            all_models = db.session.query(MLModelRegistry).all()
            
            valid_types = ['keyword', 'rule', 'hybrid', 'llm', 'multimodal', 'image']
            fixed_count = 0
            
            for model in all_models:
                if model.model_type not in valid_types:
                    # Map common invalid types to valid ones
                    type_mapping = {
                        'gpt': 'llm',
                        'claude': 'llm', 
                        'gemini': 'llm',
                        'ollama': 'llm',
                        'text': 'llm',
                        'chat': 'llm'
                    }
                    
                    # Try to map the type
                    new_type = 'llm'  # Default to llm
                    for invalid_type, valid_type in type_mapping.items():
                        if invalid_type in model.model_type.lower():
                            new_type = valid_type
                            break
                    
                    if 'dall-e' in model.id or 'image' in model.model_type:
                        new_type = 'image'
                    elif 'multimodal' in model.model_type or 'vision' in model.model_type:
                        new_type = 'multimodal'
                    
                    print(f"Fixing model {model.id}: {model.model_type} -> {new_type}")
                    model.model_type = new_type
                    fixed_count += 1
            
            db.session.commit()
            print(f"✅ Fixed {fixed_count} models")
            
        except Exception as e:
            print(f"❌ Error: {e}")
            db.session.rollback()

if __name__ == "__main__":
    cleanup_invalid_models()
