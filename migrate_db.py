#!/usr/bin/env python3
"""
Database Migration Script
Creates or updates database schema to match the new agent models
"""

import os
import sys
import sqlite3
from datetime import datetime

# Add the project root to Python path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from flask import Flask
from app import create_app
from app.extensions import db
from app.models.agent import Agent, AgentCapability, AgentSession, AgentMetrics, AgentRegistration

def create_tables():
    """Create all database tables"""
    try:
        # Create all tables
        db.create_all()
        print("✅ Database tables created successfully")
        
        # Check if tables exist
        inspector = db.inspect(db.engine)
        tables = inspector.get_table_names()
        
        expected_tables = ['agent', 'agent_capability', 'agent_session', 'agent_metrics', 'user']
        
        for table in expected_tables:
            if table in tables:
                print(f"✅ Table '{table}' exists")
            else:
                print(f"⚠️  Table '{table}' missing")
        
        return True
        
    except Exception as e:
        print(f"❌ Error creating tables: {e}")
        return False

def migrate_existing_data():
    """Migrate data from old structure to new structure if needed"""
    try:
        # Check if we need to migrate AgentRegistration data to Agent table
        with db.engine.connect() as conn:
            result = conn.execute(db.text("SELECT name FROM sqlite_master WHERE type='table' AND name='agent_registration'"))
            if result.fetchone():
                print("📦 Found existing agent_registration table, migrating data...")
                
                # Get data from old table
                old_agents = conn.execute(db.text("SELECT * FROM agent_registration")).fetchall()
                
                # Convert to new Agent records
                for old_agent in old_agents:
                    try:
                        # Create new Agent record
                        new_agent = Agent(
                            id=old_agent.id,
                            name=old_agent.name,
                            description=old_agent.description,
                            version=old_agent.version,
                            type='legacy',  # Default type
                            endpoint=old_agent.endpoint,
                            status='active' if old_agent.is_active else 'inactive',
                            is_active=old_agent.is_active,
                            is_healthy=old_agent.is_healthy,
                            total_requests=old_agent.total_requests,
                            successful_requests=old_agent.successful_requests,
                            failed_requests=old_agent.failed_requests,
                            avg_response_time=old_agent.avg_response_time,
                            last_seen=old_agent.last_seen,
                            metadata=old_agent.agent_metadata,
                            created_at=old_agent.created_at
                        )
                        
                        db.session.merge(new_agent)  # Use merge to handle duplicates
                        
                        # Create capabilities if they exist
                        if hasattr(old_agent, 'capabilities') and old_agent.capabilities:
                            capabilities = old_agent.capabilities
                            if isinstance(capabilities, dict):
                                for cap_name, confidence in capabilities.items():
                                    capability = AgentCapability(
                                        agent_id=old_agent.id,
                                        capability=cap_name,
                                        confidence_score=float(confidence) if isinstance(confidence, (int, float)) else 1.0,
                                        is_active=True,
                                        created_at=datetime.utcnow()
                                    )
                                    db.session.merge(capability)
                        
                        # Create initial metrics
                        metrics = AgentMetrics(
                            agent_id=old_agent.id,
                            agent_name=old_agent.name,
                            period_start=old_agent.created_at or datetime.utcnow(),
                            period_end=datetime.utcnow(),
                            granularity='day',
                            total_requests=old_agent.total_requests,
                            successful_requests=old_agent.successful_requests,
                            failed_requests=old_agent.failed_requests,
                            average_response_time=old_agent.avg_response_time,
                            created_at=datetime.utcnow(),
                            last_updated=datetime.utcnow()
                        )
                        db.session.merge(metrics)
                        
                    except Exception as e:
                        print(f"⚠️  Error migrating agent {old_agent.id}: {e}")
                        continue
                
                db.session.commit()
                print("✅ Data migration completed")
            else:
                print("ℹ️  No existing agent_registration table found, skipping migration")
        
        return True
        
    except Exception as e:
        print(f"❌ Error during data migration: {e}")
        db.session.rollback()
        return False

def create_sample_data():
    """Create sample agent data for testing"""
    try:
        # Check if we already have agents
        if Agent.query.count() > 0:
            print("ℹ️  Agents already exist, skipping sample data creation")
            return True
        
        print("📦 Creating sample agent data...")
        
        # Create a sample agent
        sample_agent = Agent(
            id="sample-agent-001",
            name="Sample Text Agent",
            description="A sample text processing agent for testing",
            version="1.0.0",
            type="text-processor",
            endpoint="http://localhost:8080/api/process",
            status="active",
            is_active=True,
            is_healthy=True,
            max_concurrent_sessions=5,
            metadata={"provider": "local", "model": "sample-v1"}
        )
        
        db.session.add(sample_agent)
        
        # Add capabilities
        capabilities = [
            AgentCapability(
                agent_id="sample-agent-001",
                capability="text_analysis",
                confidence_score=0.9,
                is_active=True
            ),
            AgentCapability(
                agent_id="sample-agent-001",
                capability="summarization",
                confidence_score=0.8,
                is_active=True
            )
        ]
        
        for cap in capabilities:
            db.session.add(cap)
        
        # Add initial metrics
        metrics = AgentMetrics(
            agent_id="sample-agent-001",
            agent_name="Sample Text Agent",
            period_start=datetime.utcnow(),
            period_end=datetime.utcnow(),
            granularity="day",
            total_requests=0,
            successful_requests=0,
            failed_requests=0,
            average_response_time=0.0
        )
        
        db.session.add(metrics)
        db.session.commit()
        
        print("✅ Sample data created successfully")
        return True
        
    except Exception as e:
        print(f"❌ Error creating sample data: {e}")
        db.session.rollback()
        return False

def main():
    """Main migration function"""
    print("🚀 Starting database migration...")
    
    # Create Flask app
    app = create_app('development')
    
    with app.app_context():
        # Step 1: Create tables
        if not create_tables():
            print("❌ Migration failed: Could not create tables")
            return False
        
        # Step 2: Migrate existing data
        if not migrate_existing_data():
            print("❌ Migration failed: Could not migrate data")
            return False
        
        # Step 3: Create sample data
        if not create_sample_data():
            print("⚠️  Warning: Could not create sample data")
        
        print("🎉 Database migration completed successfully!")
        return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
