-- init.sql - Database initialization script
-- This file will be executed when the PostgreSQL container starts for the first time

-- Create additional database user permissions if needed
GRANT ALL PRIVILEGES ON DATABASE ml_router_db TO ml_router_user;

-- Create extensions that might be needed
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";
CREATE EXTENSION IF NOT EXISTS "btree_gin";

-- Set timezone
SET timezone = 'UTC';

-- Create any initial data tables or indexes here if needed
-- (The Flask-SQLAlchemy models will create the actual tables)

-- Example: Create a simple health check table
CREATE TABLE IF NOT EXISTS health_check (
    id SERIAL PRIMARY KEY,
    status VARCHAR(50) NOT NULL DEFAULT 'healthy',
    last_check TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Insert initial health check record
INSERT INTO health_check (status) VALUES ('healthy') ON CONFLICT DO NOTHING;

-- Log the initialization
DO $$
BEGIN
    RAISE NOTICE 'ML Router database initialized successfully';
END $$;