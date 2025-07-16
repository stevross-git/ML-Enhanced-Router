from app import app, db
from models import QueryLog, AgentRegistration, RouterMetrics, MLModelRegistry, AICacheEntry, AICacheStats, ChatSession, ChatMessage, Document, DocumentChunk, RAGQuery


def create_all_tables():
    with app.app_context():
        db.create_all()


if __name__ == "__main__":
    create_all_tables()
