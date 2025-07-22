"""
Advanced Email Intelligence Routes
Email processing, classification, and AI-powered features
"""

from datetime import datetime
from flask import Blueprint, request, jsonify, current_app

from ..utils.decorators import rate_limit, validate_json
from ..utils.exceptions import ValidationError, ServiceError

# Create blueprint
email_advanced_bp = Blueprint('email_advanced', __name__)

def get_email_intelligence():
    """Get email intelligence instance"""
    try:
        from email_intelligence import get_email_intelligence, EmailProvider, EmailTone, EmailClassification, EmailIntent
        from ..services import get_ai_model_manager, get_personal_ai_router
        
        ai_model_manager = get_ai_model_manager()
        personal_ai_router = get_personal_ai_router()
        
        if not ai_model_manager:
            current_app.logger.warning("AI model manager not available for email intelligence")
            return None
            
        # Use memory store from personal_ai_router if available
        memory_store = None
        if personal_ai_router and hasattr(personal_ai_router, "memory_store"):
            memory_store = personal_ai_router.memory_store
        else:
            try:
                from personal_ai_router import PersonalMemoryStore
                memory_store = PersonalMemoryStore()
            except ImportError:
                current_app.logger.warning("Personal memory store not available")
        
        return get_email_intelligence(ai_model_manager, memory_store)
    except ImportError:
        current_app.logger.warning("Email intelligence not available")
        return None

@email_advanced_bp.route('/configure', methods=['POST'])
@rate_limit("10 per minute")
@validate_json(['provider', 'settings'])
def configure_email_provider():
    """Configure email provider settings"""
    try:
        email_intelligence = get_email_intelligence()
        if not email_intelligence:
            return jsonify({"error": "Email intelligence system not available"}), 503
        
        data = request.get_json()
        provider_type = data['provider']
        settings = data['settings']
        
        # Validate provider type
        from email_intelligence import EmailProvider
        if provider_type not in [p.value for p in EmailProvider]:
            return jsonify({"error": f"Invalid provider type: {provider_type}"}), 400
        
        provider = EmailProvider(provider_type)
        
        # Configure the provider
        success = email_intelligence.configure_provider(provider, settings)
        
        if success:
            return jsonify({
                "status": "success",
                "message": f"Email provider {provider_type} configured successfully",
                "provider": provider_type,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Failed to configure email provider"}), 500
        
    except Exception as e:
        current_app.logger.error(f"Error configuring email provider: {e}")
        return jsonify({"error": str(e)}), 500

@email_advanced_bp.route('/fetch', methods=['POST'])
@rate_limit("30 per minute")
@validate_json(['provider'])
def fetch_emails():
    """Fetch emails from configured provider"""
    try:
        email_intelligence = get_email_intelligence()
        if not email_intelligence:
            return jsonify({"error": "Email intelligence system not available"}), 503
        
        data = request.get_json()
        provider_type = data['provider']
        limit = data.get('limit', 50)
        folder = data.get('folder', 'INBOX')
        
        # Validate provider type
        from email_intelligence import EmailProvider
        if provider_type not in [p.value for p in EmailProvider]:
            return jsonify({"error": f"Invalid provider type: {provider_type}"}), 400
        
        provider = EmailProvider(provider_type)
        
        # Fetch emails
        messages = email_intelligence.fetch_messages(provider, limit, folder)
        
        # Convert to dict for JSON response
        messages_dict = []
        for msg in messages:
            messages_dict.append({
                "id": msg.id,
                "subject": msg.subject,
                "sender": msg.sender,
                "recipient": msg.recipient,
                "body": msg.body[:500] + "..." if len(msg.body) > 500 else msg.body,
                "timestamp": msg.timestamp.isoformat(),
                "thread_id": msg.thread_id,
                "classification": msg.classification.value if msg.classification else None,
                "intent": msg.intent.value if msg.intent else None,
                "confidence": msg.confidence,
                "extracted_entities": msg.extracted_entities,
                "action_items": msg.action_items,
                "attachments": [att.filename for att in msg.attachments] if msg.attachments else []
            })
        
        return jsonify({
            "status": "success",
            "provider": provider_type,
            "folder": folder,
            "messages": messages_dict,
            "count": len(messages_dict),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error fetching emails: {e}")
        return jsonify({"error": str(e)}), 500

@email_advanced_bp.route('/messages', methods=['GET'])
@rate_limit("100 per minute")
def get_email_messages():
    """Get stored email messages"""
    try:
        email_intelligence = get_email_intelligence()
        if not email_intelligence:
            return jsonify({"error": "Email intelligence system not available"}), 503
        
        limit = request.args.get('limit', 50, type=int)
        
        # Get messages from database
        messages = email_intelligence.db.get_messages(limit)
        
        # Convert to dict for JSON response
        messages_dict = []
        for msg in messages:
            messages_dict.append({
                "id": msg.id,
                "subject": msg.subject,
                "sender": msg.sender,
                "recipient": msg.recipient,
                "body": msg.body[:500] + "..." if len(msg.body) > 500 else msg.body,
                "timestamp": msg.timestamp.isoformat(),
                "thread_id": msg.thread_id,
                "classification": msg.classification.value if msg.classification else None,
                "intent": msg.intent.value if msg.intent else None,
                "confidence": msg.confidence,
                "extracted_entities": msg.extracted_entities,
                "action_items": msg.action_items
            })
        
        return jsonify({
            "status": "success",
            "messages": messages_dict,
            "count": len(messages_dict),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting email messages: {e}")
        return jsonify({"error": str(e)}), 500

@email_advanced_bp.route('/classify', methods=['POST'])
@rate_limit("100 per minute")
@validate_json(['email_content'])
def classify_email():
    """Classify an email"""
    try:
        email_intelligence = get_email_intelligence()
        if not email_intelligence:
            return jsonify({"error": "Email intelligence system not available"}), 503
        
        data = request.get_json()
        email_content = data['email_content']
        
        # Classify the email
        classification_result = email_intelligence.classify_email(email_content)
        
        return jsonify({
            "status": "success",
            "classification": {
                "category": classification_result.classification.value if classification_result.classification else None,
                "intent": classification_result.intent.value if classification_result.intent else None,
                "confidence": classification_result.confidence,
                "entities": classification_result.entities,
                "action_items": classification_result.action_items,
                "urgency_score": classification_result.urgency_score,
                "sentiment": classification_result.sentiment
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error classifying email: {e}")
        return jsonify({"error": str(e)}), 500

@email_advanced_bp.route('/generate-reply', methods=['POST'])
@rate_limit("20 per minute")
@validate_json(['message_id'])
def generate_email_reply():
    """Generate a reply to an email"""
    try:
        email_intelligence = get_email_intelligence()
        if not email_intelligence:
            return jsonify({"error": "Email intelligence system not available"}), 503
        
        data = request.get_json()
        message_id = data['message_id']
        tone = data.get('tone', 'professional')
        persona = data.get('persona')
        
        # Validate tone
        from email_intelligence import EmailTone
        if tone not in [t.value for t in EmailTone]:
            return jsonify({"error": f"Invalid tone: {tone}"}), 400
        
        email_tone = EmailTone(tone)
        
        # Generate reply
        reply = email_intelligence.generate_reply(message_id, email_tone, persona)
        
        if not reply:
            return jsonify({"error": "Message not found or reply generation failed"}), 404
        
        return jsonify({
            "status": "success",
            "reply": {
                "recipient": reply.recipient,
                "subject": reply.subject,
                "body": reply.body,
                "tone": reply.tone.value,
                "confidence": reply.confidence,
                "reasoning": reply.reasoning,
                "suggestions": reply.suggestions
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error generating email reply: {e}")
        return jsonify({"error": str(e)}), 500

@email_advanced_bp.route('/send', methods=['POST'])
@rate_limit("10 per minute")
@validate_json(['to', 'subject', 'body'])
def send_email():
    """Send an email through configured provider"""
    try:
        email_intelligence = get_email_intelligence()
        if not email_intelligence:
            return jsonify({"error": "Email intelligence system not available"}), 503
        
        data = request.get_json()
        to_addresses = data['to'] if isinstance(data['to'], list) else [data['to']]
        subject = data['subject']
        body = data['body']
        html_body = data.get('html_body')
        attachments = data.get('attachments', [])
        
        # Send email
        success = email_intelligence.send_email(
            to_addresses=to_addresses,
            subject=subject,
            body=body,
            html_body=html_body,
            attachments=attachments
        )
        
        if success:
            return jsonify({
                "status": "success",
                "message": "Email sent successfully",
                "recipients": to_addresses,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Failed to send email"}), 500
        
    except Exception as e:
        current_app.logger.error(f"Error sending email: {e}")
        return jsonify({"error": str(e)}), 500

@email_advanced_bp.route('/summary', methods=['GET'])
@rate_limit("50 per minute")
def get_email_summary():
    """Get email intelligence summary and statistics"""
    try:
        email_intelligence = get_email_intelligence()
        if not email_intelligence:
            return jsonify({"error": "Email intelligence system not available"}), 503
        
        summary = email_intelligence.get_summary()
        
        return jsonify({
            "status": "success",
            "summary": summary,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting email summary: {e}")
        return jsonify({"error": str(e)}), 500

@email_advanced_bp.route('/writing-style/train', methods=['POST'])
@rate_limit("5 per minute")
@validate_json(['user_id', 'email_samples'])
def train_writing_style():
    """Train AI on user's writing style"""
    try:
        email_intelligence = get_email_intelligence()
        if not email_intelligence:
            return jsonify({"error": "Email intelligence system not available"}), 503
        
        data = request.get_json()
        user_id = data['user_id']
        email_samples = data['email_samples']
        consent = data.get('consent', True)
        
        if not consent:
            return jsonify({"error": "User consent required for training"}), 400
        
        # Train on writing style
        training_id = email_intelligence.train_writing_style(user_id, email_samples)
        
        return jsonify({
            "status": "success",
            "message": "Writing style training initiated",
            "training_id": training_id,
            "user_id": user_id,
            "samples_count": len(email_samples),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error training writing style: {e}")
        return jsonify({"error": str(e)}), 500

@email_advanced_bp.route('/writing-style/approve/<training_id>', methods=['POST'])
@rate_limit("20 per minute")
def approve_training_data(training_id):
    """Approve training data for use"""
    try:
        email_intelligence = get_email_intelligence()
        if not email_intelligence:
            return jsonify({"error": "Email intelligence system not available"}), 503
        
        user_id = request.args.get('user_id', 'default')
        
        email_intelligence.db.approve_training_data(training_id, user_id)
        
        return jsonify({
            "status": "success",
            "message": "Training data approved successfully",
            "training_id": training_id,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error approving training data: {e}")
        return jsonify({"error": str(e)}), 500

@email_advanced_bp.route('/writing-style/cleanup', methods=['POST'])
@rate_limit("5 per minute")
def cleanup_training_data():
    """Clean up expired training data"""
    try:
        email_intelligence = get_email_intelligence()
        if not email_intelligence:
            return jsonify({"error": "Email intelligence system not available"}), 503
        
        email_intelligence.db.cleanup_expired_training_data()
        
        return jsonify({
            "status": "success",
            "message": "Expired training data cleaned up successfully",
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error cleaning up training data: {e}")
        return jsonify({"error": str(e)}), 500

@email_advanced_bp.route('/writing-style/consent', methods=['GET'])
@rate_limit("100 per minute")
def get_training_consent_options():
    """Get available training consent options"""
    try:
        from email_intelligence import TrainingConsent
        
        consent_options = [
            {
                "value": TrainingConsent.ENABLED.value,
                "label": "Enabled",
                "description": "Allow AI to learn from your writing style"
            },
            {
                "value": TrainingConsent.DISABLED.value,
                "label": "Disabled",
                "description": "Do not use my emails for training"
            },
            {
                "value": TrainingConsent.ASK_EACH_TIME.value,
                "label": "Ask Each Time",
                "description": "Request permission for each training opportunity"
            }
        ]
        
        return jsonify({
            "status": "success",
            "consent_options": consent_options,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting consent options: {e}")
        return jsonify({"error": str(e)}), 500

@email_advanced_bp.route('/analytics', methods=['GET'])
@rate_limit("50 per minute")
def get_email_analytics():
    """Get detailed email analytics and insights"""
    try:
        email_intelligence = get_email_intelligence()
        if not email_intelligence:
            return jsonify({"error": "Email intelligence system not available"}), 503
        
        # Get analytics data
        analytics = email_intelligence.get_analytics()
        
        return jsonify({
            "status": "success",
            "analytics": analytics,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting email analytics: {e}")
        return jsonify({"error": str(e)}), 500

@email_advanced_bp.route('/threads/<thread_id>', methods=['GET'])
@rate_limit("100 per minute")
def get_email_thread(thread_id):
    """Get all messages in an email thread"""
    try:
        email_intelligence = get_email_intelligence()
        if not email_intelligence:
            return jsonify({"error": "Email intelligence system not available"}), 503
        
        # Get thread messages
        messages = email_intelligence.get_thread_messages(thread_id)
        
        # Convert to dict for JSON response
        messages_dict = []
        for msg in messages:
            messages_dict.append({
                "id": msg.id,
                "subject": msg.subject,
                "sender": msg.sender,
                "recipient": msg.recipient,
                "body": msg.body,
                "timestamp": msg.timestamp.isoformat(),
                "thread_id": msg.thread_id,
                "classification": msg.classification.value if msg.classification else None,
                "intent": msg.intent.value if msg.intent else None,
                "confidence": msg.confidence
            })
        
        return jsonify({
            "status": "success",
            "thread_id": thread_id,
            "messages": messages_dict,
            "count": len(messages_dict),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting email thread: {e}")
        return jsonify({"error": str(e)}), 500
