"""
Peer Teaching & Collaborative Agents Routes
Agent collaboration, knowledge sharing, and peer learning systems
"""

import uuid
import sys
import subprocess
import os
from datetime import datetime
from flask import Blueprint, request, jsonify, current_app

from ..utils.decorators import rate_limit, validate_json
from ..utils.exceptions import ValidationError, ServiceError

# Create blueprint
peer_teaching_bp = Blueprint('peer_teaching', __name__)

def get_peer_teaching_system():
    """Get peer teaching system instance"""
    try:
        from peer_teaching_system import get_peer_teaching_system, AgentSpecialization, LessonType, ConsensusMethod
        return get_peer_teaching_system()
    except ImportError:
        current_app.logger.warning("Peer teaching system not available")
        return None

@peer_teaching_bp.route('/agents/register', methods=['POST'])
@rate_limit("20 per minute")
@validate_json(['agent_name', 'specialization'])
def register_peer_agent():
    """Register a new agent in the peer teaching system"""
    try:
        peer_teaching_system = get_peer_teaching_system()
        if not peer_teaching_system:
            return jsonify({"error": "Peer teaching system not available"}), 503
        
        data = request.get_json()
        agent_id = data.get('agent_id', str(uuid.uuid4()))
        agent_name = data['agent_name']
        specialization = data['specialization']
        capabilities = data.get('capabilities', [])
        
        # Validate specialization
        from peer_teaching_system import AgentSpecialization
        try:
            specialization_enum = AgentSpecialization(specialization)
        except ValueError:
            valid_specializations = [spec.value for spec in AgentSpecialization]
            return jsonify({
                "error": f"Invalid specialization. Must be one of: {valid_specializations}"
            }), 400
        
        success = peer_teaching_system.register_agent(
            agent_id, agent_name, specialization_enum, capabilities
        )
        
        if success:
            return jsonify({
                "success": True,
                "agent_id": agent_id,
                "message": f"Agent {agent_name} registered successfully",
                "specialization": specialization,
                "capabilities": capabilities,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Failed to register agent"}), 500
        
    except Exception as e:
        current_app.logger.error(f"Error registering peer agent: {e}")
        return jsonify({"error": str(e)}), 500

@peer_teaching_bp.route('/agents', methods=['GET'])
@rate_limit("100 per minute")
def get_registered_agents():
    """Get list of registered agents"""
    try:
        peer_teaching_system = get_peer_teaching_system()
        if not peer_teaching_system:
            return jsonify({"error": "Peer teaching system not available"}), 503
        
        agents = peer_teaching_system.get_registered_agents()
        
        # Convert agents to dict format
        agents_data = []
        for agent in agents:
            agents_data.append({
                "agent_id": agent.agent_id,
                "agent_name": agent.agent_name,
                "specialization": agent.specialization.value,
                "capabilities": agent.capabilities,
                "performance_score": agent.performance_score,
                "lessons_taught": agent.lessons_taught,
                "lessons_learned": agent.lessons_learned,
                "last_active": agent.last_active.isoformat() if agent.last_active else None,
                "status": agent.status
            })
        
        return jsonify({
            "registered_agents": agents_data,
            "count": len(agents_data),
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting registered agents: {e}")
        return jsonify({"error": str(e)}), 500

@peer_teaching_bp.route('/lessons/contribute', methods=['POST'])
@rate_limit("10 per minute")
@validate_json(['agent_id', 'topic', 'content', 'lesson_type'])
def contribute_lesson():
    """Contribute a lesson to the peer teaching system"""
    try:
        peer_teaching_system = get_peer_teaching_system()
        if not peer_teaching_system:
            return jsonify({"error": "Peer teaching system not available"}), 503
        
        data = request.get_json()
        agent_id = data['agent_id']
        topic = data['topic']
        content = data['content']
        lesson_type = data['lesson_type']
        difficulty_level = data.get('difficulty_level', 'intermediate')
        tags = data.get('tags', [])
        
        # Validate lesson type
        from peer_teaching_system import LessonType
        try:
            lesson_type_enum = LessonType(lesson_type)
        except ValueError:
            valid_types = [lt.value for lt in LessonType]
            return jsonify({
                "error": f"Invalid lesson type. Must be one of: {valid_types}"
            }), 400
        
        # Validate difficulty level
        valid_difficulties = ['beginner', 'intermediate', 'advanced', 'expert']
        if difficulty_level not in valid_difficulties:
            return jsonify({
                "error": f"Invalid difficulty level. Must be one of: {valid_difficulties}"
            }), 400
        
        lesson_id = peer_teaching_system.contribute_lesson(
            agent_id=agent_id,
            topic=topic,
            content=content,
            lesson_type=lesson_type_enum,
            difficulty_level=difficulty_level,
            tags=tags
        )
        
        if lesson_id:
            return jsonify({
                "success": True,
                "lesson_id": lesson_id,
                "message": f"Lesson on '{topic}' contributed successfully",
                "agent_id": agent_id,
                "topic": topic,
                "lesson_type": lesson_type,
                "difficulty_level": difficulty_level,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Failed to contribute lesson"}), 500
        
    except Exception as e:
        current_app.logger.error(f"Error contributing lesson: {e}")
        return jsonify({"error": str(e)}), 500

@peer_teaching_bp.route('/lessons', methods=['GET'])
@rate_limit("100 per minute")
def get_available_lessons():
    """Get available lessons in the system"""
    try:
        peer_teaching_system = get_peer_teaching_system()
        if not peer_teaching_system:
            return jsonify({"error": "Peer teaching system not available"}), 503
        
        # Get query parameters
        topic = request.args.get('topic')
        lesson_type = request.args.get('type')
        difficulty = request.args.get('difficulty')
        limit = request.args.get('limit', 50, type=int)
        
        lessons = peer_teaching_system.get_available_lessons(
            topic=topic,
            lesson_type=lesson_type,
            difficulty_level=difficulty,
            limit=limit
        )
        
        # Convert lessons to dict format
        lessons_data = []
        for lesson in lessons:
            lessons_data.append({
                "lesson_id": lesson.lesson_id,
                "topic": lesson.topic,
                "content": lesson.content[:200] + "..." if len(lesson.content) > 200 else lesson.content,
                "lesson_type": lesson.lesson_type.value,
                "difficulty_level": lesson.difficulty_level,
                "tags": lesson.tags,
                "contributor_id": lesson.contributor_id,
                "rating": lesson.rating,
                "usage_count": lesson.usage_count,
                "created_at": lesson.created_at.isoformat(),
                "last_updated": lesson.last_updated.isoformat() if lesson.last_updated else None
            })
        
        return jsonify({
            "available_lessons": lessons_data,
            "count": len(lessons_data),
            "filters": {
                "topic": topic,
                "lesson_type": lesson_type,
                "difficulty": difficulty
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting available lessons: {e}")
        return jsonify({"error": str(e)}), 500

@peer_teaching_bp.route('/lessons/<lesson_id>', methods=['GET'])
@rate_limit("100 per minute")
def get_lesson_details(lesson_id):
    """Get detailed information about a specific lesson"""
    try:
        peer_teaching_system = get_peer_teaching_system()
        if not peer_teaching_system:
            return jsonify({"error": "Peer teaching system not available"}), 503
        
        lesson = peer_teaching_system.get_lesson(lesson_id)
        
        if not lesson:
            return jsonify({"error": "Lesson not found"}), 404
        
        return jsonify({
            "lesson": {
                "lesson_id": lesson.lesson_id,
                "topic": lesson.topic,
                "content": lesson.content,
                "lesson_type": lesson.lesson_type.value,
                "difficulty_level": lesson.difficulty_level,
                "tags": lesson.tags,
                "contributor_id": lesson.contributor_id,
                "rating": lesson.rating,
                "usage_count": lesson.usage_count,
                "feedback": lesson.feedback,
                "created_at": lesson.created_at.isoformat(),
                "last_updated": lesson.last_updated.isoformat() if lesson.last_updated else None
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting lesson details: {e}")
        return jsonify({"error": str(e)}), 500

@peer_teaching_bp.route('/consensus/vote', methods=['POST'])
@rate_limit("50 per minute")
@validate_json(['consensus_id', 'agent_id', 'vote'])
def vote_on_consensus():
    """Vote on a consensus decision"""
    try:
        peer_teaching_system = get_peer_teaching_system()
        if not peer_teaching_system:
            return jsonify({"error": "Peer teaching system not available"}), 503
        
        data = request.get_json()
        consensus_id = data['consensus_id']
        agent_id = data['agent_id']
        vote = data['vote']
        reasoning = data.get('reasoning', '')
        
        # Validate vote
        valid_votes = ['approve', 'reject', 'abstain']
        if vote not in valid_votes:
            return jsonify({
                "error": f"Invalid vote. Must be one of: {valid_votes}"
            }), 400
        
        success = peer_teaching_system.vote_on_consensus(
            consensus_id=consensus_id,
            agent_id=agent_id,
            vote=vote,
            reasoning=reasoning
        )
        
        if success:
            return jsonify({
                "success": True,
                "message": "Vote recorded successfully",
                "consensus_id": consensus_id,
                "agent_id": agent_id,
                "vote": vote,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Failed to record vote"}), 500
        
    except Exception as e:
        current_app.logger.error(f"Error voting on consensus: {e}")
        return jsonify({"error": str(e)}), 500

@peer_teaching_bp.route('/consensus/<consensus_id>', methods=['GET'])
@rate_limit("100 per minute")
def get_consensus_status(consensus_id):
    """Get status of a consensus decision"""
    try:
        peer_teaching_system = get_peer_teaching_system()
        if not peer_teaching_system:
            return jsonify({"error": "Peer teaching system not available"}), 503
        
        consensus = peer_teaching_system.get_consensus_status(consensus_id)
        
        if not consensus:
            return jsonify({"error": "Consensus not found"}), 404
        
        return jsonify({
            "consensus": {
                "consensus_id": consensus.consensus_id,
                "topic": consensus.topic,
                "description": consensus.description,
                "status": consensus.status,
                "method": consensus.method.value,
                "votes": consensus.votes,
                "current_result": consensus.current_result,
                "threshold": consensus.threshold,
                "deadline": consensus.deadline.isoformat() if consensus.deadline else None,
                "created_at": consensus.created_at.isoformat(),
                "resolved_at": consensus.resolved_at.isoformat() if consensus.resolved_at else None
            },
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting consensus status: {e}")
        return jsonify({"error": str(e)}), 500

@peer_teaching_bp.route('/demo', methods=['POST'])
@rate_limit("5 per minute")
def run_peer_teaching_demo():
    """Run the peer teaching demonstration"""
    try:
        # Run the simple demo script
        result = subprocess.run(
            [sys.executable, 'simple_peer_demo.py'],
            capture_output=True,
            text=True,
            cwd=os.getcwd()
        )
        
        if result.returncode == 0:
            return jsonify({
                "status": "success",
                "output": result.stdout,
                "message": "Peer teaching demo completed successfully",
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({
                "status": "error",
                "error": f"Demo failed: {result.stderr}",
                "output": result.stdout,
                "timestamp": datetime.now().isoformat()
            }), 500
            
    except Exception as e:
        current_app.logger.error(f"Error running peer teaching demo: {e}")
        return jsonify({"error": str(e)}), 500

@peer_teaching_bp.route('/stats', methods=['GET'])
@rate_limit("100 per minute")
def get_peer_teaching_stats():
    """Get peer teaching system statistics"""
    try:
        peer_teaching_system = get_peer_teaching_system()
        if not peer_teaching_system:
            return jsonify({"error": "Peer teaching system not available"}), 503
        
        stats = peer_teaching_system.get_peer_teaching_stats()
        
        return jsonify({
            "peer_teaching_stats": stats,
            "timestamp": datetime.now().isoformat()
        })
        
    except Exception as e:
        current_app.logger.error(f"Error getting peer teaching stats: {e}")
        return jsonify({"error": str(e)}), 500

@peer_teaching_bp.route('/knowledge/sync', methods=['POST'])
@rate_limit("10 per minute")
@validate_json(['source_agent', 'target_agent'])
def sync_knowledge():
    """Synchronize knowledge between agents"""
    try:
        peer_teaching_system = get_peer_teaching_system()
        if not peer_teaching_system:
            return jsonify({"error": "Peer teaching system not available"}), 503
        
        data = request.get_json()
        source_agent = data['source_agent']
        target_agent = data['target_agent']
        knowledge_areas = data.get('knowledge_areas', [])
        
        sync_id = peer_teaching_system.sync_knowledge(
            source_agent=source_agent,
            target_agent=target_agent,
            knowledge_areas=knowledge_areas
        )
        
        if sync_id:
            return jsonify({
                "success": True,
                "sync_id": sync_id,
                "message": "Knowledge synchronization initiated",
                "source_agent": source_agent,
                "target_agent": target_agent,
                "knowledge_areas": knowledge_areas,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Failed to initiate knowledge sync"}), 500
        
    except Exception as e:
        current_app.logger.error(f"Error syncing knowledge: {e}")
        return jsonify({"error": str(e)}), 500

@peer_teaching_bp.route('/collaboration/request', methods=['POST'])
@rate_limit("20 per minute")
@validate_json(['requesting_agent', 'collaboration_type'])
def request_collaboration():
    """Request collaboration between agents"""
    try:
        peer_teaching_system = get_peer_teaching_system()
        if not peer_teaching_system:
            return jsonify({"error": "Peer teaching system not available"}), 503
        
        data = request.get_json()
        requesting_agent = data['requesting_agent']
        collaboration_type = data['collaboration_type']
        target_agents = data.get('target_agents', [])
        description = data.get('description', '')
        duration = data.get('duration', 3600)  # Default 1 hour
        
        # Validate collaboration type
        valid_types = ['knowledge_share', 'problem_solve', 'lesson_creation', 'peer_review']
        if collaboration_type not in valid_types:
            return jsonify({
                "error": f"Invalid collaboration type. Must be one of: {valid_types}"
            }), 400
        
        collaboration_id = peer_teaching_system.request_collaboration(
            requesting_agent=requesting_agent,
            collaboration_type=collaboration_type,
            target_agents=target_agents,
            description=description,
            duration=duration
        )
        
        if collaboration_id:
            return jsonify({
                "success": True,
                "collaboration_id": collaboration_id,
                "message": "Collaboration request created",
                "requesting_agent": requesting_agent,
                "collaboration_type": collaboration_type,
                "target_agents": target_agents,
                "timestamp": datetime.now().isoformat()
            })
        else:
            return jsonify({"error": "Failed to create collaboration request"}), 500
        
    except Exception as e:
        current_app.logger.error(f"Error requesting collaboration: {e}")
        return jsonify({"error": str(e)}), 500
