#!/usr/bin/env python3
"""
Database Indexes Setup - Quick Win #1
Creates all necessary indexes for optimal query performance
"""

import sqlite3
import logging
from typing import List, Dict, Any
from datetime import datetime
import os

logger = logging.getLogger(__name__)

class DatabaseIndexManager:
    """Manages database indexes for optimal performance"""
    
    def __init__(self, db_path: str = "ml_router.db"):
        self.db_path = db_path
        self.indexes = self._get_index_definitions()
        
    def _get_index_definitions(self) -> List[Dict[str, str]]:
        """Get all index definitions for the database"""
        return [
            # Memory table indexes
            {
                "name": "idx_memory_user_timestamp",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_memory_user_timestamp 
                ON memory(user_id, timestamp DESC)
                """,
                "description": "Optimize memory queries by user and timestamp"
            },
            {
                "name": "idx_memory_category",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_memory_category 
                ON memory(category)
                """,
                "description": "Optimize memory queries by category"
            },
            {
                "name": "idx_memory_confidence",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_memory_confidence 
                ON memory(confidence DESC)
                """,
                "description": "Optimize memory queries by confidence score"
            },
            {
                "name": "idx_memory_user_category",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_memory_user_category 
                ON memory(user_id, category)
                """,
                "description": "Optimize memory queries by user and category"
            },
            {
                "name": "idx_memory_tags",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_memory_tags 
                ON memory(tags)
                """,
                "description": "Optimize memory queries by tags"
            },
            
            # Cache table indexes
            {
                "name": "idx_cache_key_timestamp",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_cache_key_timestamp 
                ON cache_entries(cache_key, timestamp DESC)
                """,
                "description": "Optimize cache lookups by key and timestamp"
            },
            {
                "name": "idx_cache_model_timestamp",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_cache_model_timestamp 
                ON cache_entries(model_id, timestamp DESC)
                """,
                "description": "Optimize cache queries by model and timestamp"
            },
            {
                "name": "idx_cache_expiry",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_cache_expiry 
                ON cache_entries(expires_at)
                """,
                "description": "Optimize cache expiry cleanup"
            },
            {
                "name": "idx_cache_hit_count",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_cache_hit_count 
                ON cache_entries(hit_count DESC)
                """,
                "description": "Optimize cache analytics by hit count"
            },
            
            # Classification table indexes
            {
                "name": "idx_classifications_query_hash",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_classifications_query_hash 
                ON classifications(query_hash)
                """,
                "description": "Optimize classification lookups by query hash"
            },
            {
                "name": "idx_classifications_category",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_classifications_category 
                ON classifications(category)
                """,
                "description": "Optimize classification queries by category"
            },
            {
                "name": "idx_classifications_confidence",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_classifications_confidence 
                ON classifications(confidence DESC)
                """,
                "description": "Optimize classification queries by confidence"
            },
            {
                "name": "idx_classifications_timestamp",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_classifications_timestamp 
                ON classifications(timestamp DESC)
                """,
                "description": "Optimize classification queries by timestamp"
            },
            
            # Decisions table indexes (for cognitive debugging)
            {
                "name": "idx_decisions_session_timestamp",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_decisions_session_timestamp 
                ON decisions(session_id, timestamp DESC)
                """,
                "description": "Optimize decision queries by session and timestamp"
            },
            {
                "name": "idx_decisions_user_timestamp",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_decisions_user_timestamp 
                ON decisions(user_id, timestamp DESC)
                """,
                "description": "Optimize decision queries by user and timestamp"
            },
            {
                "name": "idx_decisions_type",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_decisions_type 
                ON decisions(decision_type)
                """,
                "description": "Optimize decision queries by type"
            },
            {
                "name": "idx_decisions_confidence",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_decisions_confidence 
                ON decisions(confidence DESC)
                """,
                "description": "Optimize decision queries by confidence"
            },
            
            # Personas table indexes
            {
                "name": "idx_personas_user_type",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_personas_user_type 
                ON personas(user_id, type)
                """,
                "description": "Optimize persona queries by user and type"
            },
            {
                "name": "idx_personas_active",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_personas_active 
                ON personas(is_active)
                """,
                "description": "Optimize active persona queries"
            },
            {
                "name": "idx_personas_created_at",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_personas_created_at 
                ON personas(created_at DESC)
                """,
                "description": "Optimize persona queries by creation date"
            },
            
            # RAG documents table indexes
            {
                "name": "idx_rag_documents_type",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_rag_documents_type 
                ON rag_documents(file_type)
                """,
                "description": "Optimize RAG document queries by type"
            },
            {
                "name": "idx_rag_documents_status",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_rag_documents_status 
                ON rag_documents(status)
                """,
                "description": "Optimize RAG document queries by status"
            },
            {
                "name": "idx_rag_documents_uploaded_at",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_rag_documents_uploaded_at 
                ON rag_documents(uploaded_at DESC)
                """,
                "description": "Optimize RAG document queries by upload date"
            },
            {
                "name": "idx_rag_documents_size",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_rag_documents_size 
                ON rag_documents(file_size)
                """,
                "description": "Optimize RAG document queries by size"
            },
            
            # RAG chunks table indexes
            {
                "name": "idx_rag_chunks_document_id",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_rag_chunks_document_id 
                ON rag_chunks(document_id)
                """,
                "description": "Optimize RAG chunk queries by document"
            },
            {
                "name": "idx_rag_chunks_chunk_index",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_rag_chunks_chunk_index 
                ON rag_chunks(document_id, chunk_index)
                """,
                "description": "Optimize RAG chunk queries by document and index"
            },
            
            # Users table indexes
            {
                "name": "idx_users_email",
                "sql": """
                CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email 
                ON users(email)
                """,
                "description": "Optimize user lookups by email"
            },
            {
                "name": "idx_users_created_at",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_users_created_at 
                ON users(created_at DESC)
                """,
                "description": "Optimize user queries by creation date"
            },
            {
                "name": "idx_users_last_login",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_users_last_login 
                ON users(last_login DESC)
                """,
                "description": "Optimize user queries by last login"
            },
            
            # Sessions table indexes
            {
                "name": "idx_sessions_user_id",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_sessions_user_id 
                ON sessions(user_id)
                """,
                "description": "Optimize session queries by user"
            },
            {
                "name": "idx_sessions_created_at",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_sessions_created_at 
                ON sessions(created_at DESC)
                """,
                "description": "Optimize session queries by creation date"
            },
            {
                "name": "idx_sessions_expires_at",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_sessions_expires_at 
                ON sessions(expires_at)
                """,
                "description": "Optimize session expiry cleanup"
            },
            
            # Query logs table indexes
            {
                "name": "idx_query_logs_timestamp",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_query_logs_timestamp 
                ON query_logs(timestamp DESC)
                """,
                "description": "Optimize query log queries by timestamp"
            },
            {
                "name": "idx_query_logs_user_timestamp",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_query_logs_user_timestamp 
                ON query_logs(user_id, timestamp DESC)
                """,
                "description": "Optimize query log queries by user and timestamp"
            },
            {
                "name": "idx_query_logs_response_time",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_query_logs_response_time 
                ON query_logs(response_time DESC)
                """,
                "description": "Optimize query log queries by response time"
            },
            {
                "name": "idx_query_logs_status",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_query_logs_status 
                ON query_logs(status)
                """,
                "description": "Optimize query log queries by status"
            },
            
            # Model usage table indexes
            {
                "name": "idx_model_usage_model_timestamp",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_model_usage_model_timestamp 
                ON model_usage(model_id, timestamp DESC)
                """,
                "description": "Optimize model usage queries by model and timestamp"
            },
            {
                "name": "idx_model_usage_cost",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_model_usage_cost 
                ON model_usage(cost DESC)
                """,
                "description": "Optimize model usage queries by cost"
            },
            {
                "name": "idx_model_usage_tokens",
                "sql": """
                CREATE INDEX IF NOT EXISTS idx_model_usage_tokens 
                ON model_usage(tokens_used DESC)
                """,
                "description": "Optimize model usage queries by tokens used"
            }
        ]
    
    def create_all_indexes(self) -> Dict[str, Any]:
        """Create all indexes and return results"""
        results = {
            "created": [],
            "skipped": [],
            "errors": [],
            "total_time": 0
        }
        
        start_time = datetime.now()
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for index_def in self.indexes:
                    try:
                        index_start = datetime.now()
                        
                        # Check if index already exists
                        cursor.execute("""
                            SELECT name FROM sqlite_master 
                            WHERE type='index' AND name=?
                        """, (index_def["name"],))
                        
                        if cursor.fetchone():
                            results["skipped"].append({
                                "name": index_def["name"],
                                "reason": "Index already exists"
                            })
                            continue
                        
                        # Create the index
                        cursor.execute(index_def["sql"])
                        
                        index_end = datetime.now()
                        index_time = (index_end - index_start).total_seconds()
                        
                        results["created"].append({
                            "name": index_def["name"],
                            "description": index_def["description"],
                            "creation_time": index_time
                        })
                        
                        logger.info(f"Created index {index_def['name']} in {index_time:.3f}s")
                        
                    except Exception as e:
                        error_msg = f"Failed to create index {index_def['name']}: {str(e)}"
                        results["errors"].append({
                            "name": index_def["name"],
                            "error": error_msg
                        })
                        logger.error(error_msg)
                
                conn.commit()
                
        except Exception as e:
            error_msg = f"Database connection error: {str(e)}"
            results["errors"].append({"general": error_msg})
            logger.error(error_msg)
        
        end_time = datetime.now()
        results["total_time"] = (end_time - start_time).total_seconds()
        
        return results
    
    def drop_all_indexes(self) -> Dict[str, Any]:
        """Drop all indexes (for testing or rebuilding)"""
        results = {
            "dropped": [],
            "not_found": [],
            "errors": []
        }
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for index_def in self.indexes:
                    try:
                        # Check if index exists
                        cursor.execute("""
                            SELECT name FROM sqlite_master 
                            WHERE type='index' AND name=?
                        """, (index_def["name"],))
                        
                        if not cursor.fetchone():
                            results["not_found"].append(index_def["name"])
                            continue
                        
                        # Drop the index
                        cursor.execute(f"DROP INDEX IF EXISTS {index_def['name']}")
                        results["dropped"].append(index_def["name"])
                        
                        logger.info(f"Dropped index {index_def['name']}")
                        
                    except Exception as e:
                        error_msg = f"Failed to drop index {index_def['name']}: {str(e)}"
                        results["errors"].append({
                            "name": index_def["name"],
                            "error": error_msg
                        })
                        logger.error(error_msg)
                
                conn.commit()
                
        except Exception as e:
            error_msg = f"Database connection error: {str(e)}"
            results["errors"].append({"general": error_msg})
            logger.error(error_msg)
        
        return results
    
    def analyze_index_usage(self) -> Dict[str, Any]:
        """Analyze index usage statistics"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Get all indexes
                cursor.execute("""
                    SELECT name, sql FROM sqlite_master 
                    WHERE type='index' AND name NOT LIKE 'sqlite_%'
                    ORDER BY name
                """)
                
                indexes = cursor.fetchall()
                
                # Get database size
                cursor.execute("SELECT page_count * page_size as size FROM pragma_page_count(), pragma_page_size()")
                db_size = cursor.fetchone()[0]
                
                # Get table row counts
                cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
                tables = cursor.fetchall()
                
                table_stats = {}
                for table in tables:
                    table_name = table[0]
                    try:
                        cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                        row_count = cursor.fetchone()[0]
                        table_stats[table_name] = row_count
                    except:
                        table_stats[table_name] = 0
                
                return {
                    "total_indexes": len(indexes),
                    "database_size_bytes": db_size,
                    "database_size_mb": db_size / (1024 * 1024),
                    "table_row_counts": table_stats,
                    "indexes": [{"name": idx[0], "sql": idx[1]} for idx in indexes]
                }
                
        except Exception as e:
            logger.error(f"Error analyzing index usage: {str(e)}")
            return {"error": str(e)}
    
    def get_query_performance_test(self) -> Dict[str, Any]:
        """Test query performance with and without indexes"""
        test_queries = [
            {
                "name": "memory_by_user_and_timestamp",
                "sql": "SELECT * FROM memory WHERE user_id = ? ORDER BY timestamp DESC LIMIT 10",
                "params": ("test_user_123",)
            },
            {
                "name": "cache_by_key",
                "sql": "SELECT * FROM cache_entries WHERE cache_key = ? AND expires_at > ?",
                "params": ("test_key", datetime.now())
            },
            {
                "name": "classifications_by_category",
                "sql": "SELECT * FROM classifications WHERE category = ? ORDER BY confidence DESC",
                "params": ("technical",)
            },
            {
                "name": "decisions_by_session",
                "sql": "SELECT * FROM decisions WHERE session_id = ? ORDER BY timestamp DESC",
                "params": ("test_session",)
            }
        ]
        
        results = {}
        
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                for query in test_queries:
                    try:
                        # Enable query planner
                        cursor.execute("EXPLAIN QUERY PLAN " + query["sql"], query["params"])
                        query_plan = cursor.fetchall()
                        
                        # Time the query
                        start_time = datetime.now()
                        cursor.execute(query["sql"], query["params"])
                        cursor.fetchall()
                        end_time = datetime.now()
                        
                        execution_time = (end_time - start_time).total_seconds() * 1000  # Convert to ms
                        
                        results[query["name"]] = {
                            "execution_time_ms": execution_time,
                            "query_plan": query_plan,
                            "uses_index": any("USING INDEX" in str(step) for step in query_plan)
                        }
                        
                    except Exception as e:
                        results[query["name"]] = {"error": str(e)}
                        
        except Exception as e:
            return {"error": f"Database connection error: {str(e)}"}
        
        return results


def main():
    """Main function to create indexes"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Database Index Manager")
    parser.add_argument("--db-path", default="ml_router.db", help="Database file path")
    parser.add_argument("--action", choices=["create", "drop", "analyze", "test"], 
                       default="create", help="Action to perform")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Initialize manager
    manager = DatabaseIndexManager(args.db_path)
    
    if args.action == "create":
        print("Creating database indexes...")
        results = manager.create_all_indexes()
        
        print(f"\nResults:")
        print(f"Created: {len(results['created'])} indexes")
        print(f"Skipped: {len(results['skipped'])} indexes")
        print(f"Errors: {len(results['errors'])} indexes")
        print(f"Total time: {results['total_time']:.3f} seconds")
        
        if results['errors']:
            print("\nErrors:")
            for error in results['errors']:
                print(f"  - {error}")
        
        if args.verbose and results['created']:
            print("\nCreated indexes:")
            for created in results['created']:
                print(f"  - {created['name']}: {created['description']} ({created['creation_time']:.3f}s)")
    
    elif args.action == "drop":
        print("Dropping database indexes...")
        results = manager.drop_all_indexes()
        
        print(f"\nResults:")
        print(f"Dropped: {len(results['dropped'])} indexes")
        print(f"Not found: {len(results['not_found'])} indexes")
        print(f"Errors: {len(results['errors'])} indexes")
    
    elif args.action == "analyze":
        print("Analyzing database indexes...")
        results = manager.analyze_index_usage()
        
        if "error" in results:
            print(f"Error: {results['error']}")
        else:
            print(f"\nDatabase Analysis:")
            print(f"Total indexes: {results['total_indexes']}")
            print(f"Database size: {results['database_size_mb']:.2f} MB")
            print(f"Table row counts:")
            for table, count in results['table_row_counts'].items():
                print(f"  - {table}: {count:,} rows")
    
    elif args.action == "test":
        print("Testing query performance...")
        results = manager.get_query_performance_test()
        
        print(f"\nQuery Performance Test Results:")
        for query_name, result in results.items():
            if "error" in result:
                print(f"  {query_name}: ERROR - {result['error']}")
            else:
                index_status = "✓ USING INDEX" if result['uses_index'] else "✗ NO INDEX"
                print(f"  {query_name}: {result['execution_time_ms']:.2f}ms {index_status}")


if __name__ == "__main__":
    main()
