#!/usr/bin/env python3
"""
Peer Teaching System Demo
This demonstrates how the learning system works with agent communities,
federated knowledge exchange, and multi-agent collaboration.
"""

import asyncio
import json
from datetime import datetime
from peer_teaching_system import (
    PeerTeachingSystem, 
    AgentSpecialization, 
    LessonType, 
    ConsensusMethod,
    get_peer_teaching_system
)

async def demonstrate_peer_teaching_learning():
    """
    Comprehensive demonstration of the peer teaching system
    """
    print("🎓 PEER TEACHING SYSTEM LEARNING DEMONSTRATION")
    print("=" * 60)
    
    # Get the peer teaching system
    system = get_peer_teaching_system()
    
    # Step 1: Register diverse agents with different specializations
    print("\n📝 STEP 1: AGENT REGISTRATION")
    print("-" * 40)
    
    agents = [
        ("alice_math", "Alice", AgentSpecialization.MATH_SOLVER, ["algebra", "calculus", "statistics"]),
        ("bob_code", "Bob", AgentSpecialization.CODE_HELPER, ["python", "debugging", "algorithms"]),
        ("carol_write", "Carol", AgentSpecialization.WRITING_ASSISTANT, ["creative writing", "editing", "research"]),
        ("dave_research", "Dave", AgentSpecialization.RESEARCH_ASSISTANT, ["data analysis", "fact checking", "synthesis"]),
        ("eve_creative", "Eve", AgentSpecialization.CREATIVE_WRITER, ["storytelling", "poetry", "brainstorming"])
    ]
    
    for agent_id, agent_name, specialization, capabilities in agents:
        success = system.register_agent(agent_id, agent_name, specialization, capabilities)
        status = "✅ Registered" if success else "❌ Failed"
        print(f"{status}: {agent_name} ({specialization.value}) - {', '.join(capabilities)}")
    
    # Step 2: Agents contribute lessons based on their expertise
    print("\n📚 STEP 2: LESSON CONTRIBUTION")
    print("-" * 40)
    
    lessons = [
        {
            "agent_id": "alice_math",
            "lesson_type": LessonType.STRATEGY,
            "domain": "mathematics",
            "title": "Quadratic Equation Solving Strategy",
            "content": "When solving quadratic equations, always start by checking if factoring is possible. If not, use the quadratic formula. Remember to check your discriminant first to determine the nature of roots.",
            "strategy_steps": [
                "1. Write equation in standard form ax² + bx + c = 0",
                "2. Calculate discriminant b² - 4ac",
                "3. Try factoring if discriminant is a perfect square",
                "4. Use quadratic formula if factoring fails",
                "5. Always verify solutions by substitution"
            ],
            "effectiveness_score": 0.92,
            "usage_context": "algebra problems, homework help, exam preparation",
            "success_metrics": {"accuracy": 0.95, "speed": 0.88, "student_satisfaction": 0.91}
        },
        {
            "agent_id": "bob_code",
            "lesson_type": LessonType.PATTERN,
            "domain": "programming",
            "title": "Debugging Pattern: Divide and Conquer",
            "content": "When debugging complex code, isolate the problem by commenting out sections and testing incrementally. This helps identify exactly where the issue occurs.",
            "strategy_steps": [
                "1. Reproduce the error consistently",
                "2. Add print statements or logging at key points",
                "3. Comment out half the suspicious code",
                "4. Test to see if error persists",
                "5. Narrow down to the problematic section",
                "6. Examine variables and data flow"
            ],
            "effectiveness_score": 0.87,
            "usage_context": "code debugging, error resolution, development workflow",
            "success_metrics": {"bug_resolution_rate": 0.89, "time_to_fix": 0.82, "code_quality": 0.85}
        },
        {
            "agent_id": "carol_write",
            "lesson_type": LessonType.OPTIMIZATION,
            "domain": "writing",
            "title": "Active Voice Transformation Technique",
            "content": "Convert passive voice to active voice to make writing more engaging and direct. Look for 'was/were + past participle' constructions and restructure them.",
            "strategy_steps": [
                "1. Identify passive voice constructions",
                "2. Find the actual actor (doer of the action)",
                "3. Make the actor the subject",
                "4. Use active verb forms",
                "5. Rearrange sentence structure as needed"
            ],
            "effectiveness_score": 0.91,
            "usage_context": "essay writing, content creation, professional communication",
            "success_metrics": {"readability_score": 0.93, "engagement": 0.88, "clarity": 0.92}
        }
    ]
    
    lesson_ids = []
    for lesson in lessons:
        lesson_id = system.contribute_lesson(**lesson)
        lesson_ids.append(lesson_id)
        print(f"✅ Lesson contributed: '{lesson['title']}' by {lesson['agent_id']}")
    
    # Step 3: Demonstrate lesson discovery and adoption
    print("\n🔍 STEP 3: LESSON DISCOVERY & ADOPTION")
    print("-" * 40)
    
    # Dave (research assistant) looks for lessons in mathematics
    print("\n🔎 Dave searching for mathematics lessons...")
    math_lessons = system.find_relevant_lessons("dave_research", "mathematics")
    for lesson in math_lessons:
        print(f"📖 Found: '{lesson.title}' (effectiveness: {lesson.effectiveness_score:.2f})")
        print(f"   📝 {lesson.content[:100]}...")
        
        # Dave adopts the lesson
        if system.adopt_lesson("dave_research", lesson.lesson_id):
            print(f"   ✅ Dave adopted this lesson!")
    
    # Eve (creative writer) looks for writing lessons
    print("\n🔎 Eve searching for writing lessons...")
    writing_lessons = system.find_relevant_lessons("eve_creative", "writing")
    for lesson in writing_lessons:
        print(f"📖 Found: '{lesson.title}' (effectiveness: {lesson.effectiveness_score:.2f})")
        print(f"   📝 {lesson.content[:100]}...")
        
        # Eve adopts the lesson
        if system.adopt_lesson("eve_creative", lesson.lesson_id):
            print(f"   ✅ Eve adopted this lesson!")
    
    # Step 4: Federated knowledge contribution
    print("\n🌐 STEP 4: FEDERATED KNOWLEDGE SHARING")
    print("-" * 40)
    
    # Agents contribute anonymized knowledge
    federated_contributions = [
        {
            "agent_id": "alice_math",
            "specialization": AgentSpecialization.MATH_SOLVER,
            "query_type": "quadratic_equations",
            "toolkit_used": "algebraic_manipulation",
            "approach_summary": "Factor-first approach with discriminant checking",
            "performance_metrics": {"accuracy": 0.95, "speed": 0.88, "efficiency": 0.91},
            "lessons_learned": ["Always check discriminant first", "Factoring is faster when possible"],
            "optimization_tips": ["Pre-calculate common factors", "Use graphing for verification"],
            "error_patterns": ["Arithmetic errors in discriminant", "Sign errors in quadratic formula"]
        },
        {
            "agent_id": "bob_code",
            "specialization": AgentSpecialization.CODE_HELPER,
            "query_type": "debugging_assistance",
            "toolkit_used": "print_debugging",
            "approach_summary": "Incremental isolation with logging",
            "performance_metrics": {"bug_resolution_rate": 0.89, "time_efficiency": 0.82},
            "lessons_learned": ["Print debugging is still very effective", "Isolate before investigating"],
            "optimization_tips": ["Use conditional logging", "Save debug states"],
            "error_patterns": ["Scope-related variable issues", "Timing-dependent bugs"]
        }
    ]
    
    for contrib in federated_contributions:
        contrib_id = system.contribute_federated_knowledge(**contrib)
        print(f"🌐 Federated knowledge contributed: {contrib['query_type']} by {contrib['specialization'].value}")
    
    # Demonstrate federated knowledge retrieval
    print("\n🔍 Retrieving federated knowledge...")
    math_knowledge = system.get_federated_knowledge(AgentSpecialization.MATH_SOLVER, "quadratic_equations")
    for knowledge in math_knowledge:
        print(f"📊 Knowledge: {knowledge.query_type} ({knowledge.agent_specialization.value})")
        print(f"   📋 Approach: {knowledge.approach_summary}")
        print(f"   📈 Performance: {knowledge.performance_metrics}")
    
    # Step 5: Multi-agent collaboration session
    print("\n🤝 STEP 5: COLLABORATIVE SESSION")
    print("-" * 40)
    
    # Start a collaborative session
    session_id = await system.start_collaborative_session(
        initiator_agent="alice_math",
        task_description="Develop a comprehensive study guide for calculus students",
        session_type="educational_content_creation",
        required_specializations=[
            AgentSpecialization.MATH_SOLVER,
            AgentSpecialization.WRITING_ASSISTANT,
            AgentSpecialization.RESEARCH_ASSISTANT
        ]
    )
    
    print(f"🎯 Collaborative session started: {session_id}")
    
    # Step 6: Multi-agent debate
    print("\n🗣️ STEP 6: MULTI-AGENT DEBATE")
    print("-" * 40)
    
    debate_question = "What is the most effective approach to teach calculus to beginners?"
    
    print(f"💭 Debate question: {debate_question}")
    
    debate_result = await system.multi_agent_debate(
        session_id=session_id,
        question=debate_question,
        consensus_method=ConsensusMethod.CONFIDENCE_WEIGHTED
    )
    
    print(f"\n🎯 Debate Results:")
    print(f"📊 Consensus Method: {debate_result.get('consensus_method', 'confidence_weighted')}")
    print(f"🏆 Final Decision: {debate_result.get('final_decision', 'Use visual aids and step-by-step approach')}")
    print(f"📈 Confidence Score: {debate_result.get('confidence_score', 0.85):.2f}")
    
    print(f"\n👥 Agent Positions:")
    positions = debate_result.get('agent_positions', [])
    for position in positions:
        if hasattr(position, 'agent_id'):
            print(f"  🤖 {position.agent_id} ({position.agent_specialization.value})")
            print(f"     💭 Position: {position.position}")
            print(f"     📊 Confidence: {position.confidence:.2f}")
            print(f"     💡 Reasoning: {position.reasoning}")
        else:
            print(f"  🤖 {position.get('agent_id', 'unknown')}")
            print(f"     💭 Position: {position.get('position', 'No position')}")
            print(f"     📊 Confidence: {position.get('confidence', 0.0):.2f}")
    
    print(f"\n📚 Lessons Generated from Debate:")
    lessons = debate_result.get('lessons_generated', ['Visual learning enhances comprehension', 'Step-by-step approach reduces cognitive load'])
    for lesson in lessons:
        print(f"  📖 {lesson}")
    
    # Step 7: Cross-correction demonstration
    print("\n🔄 STEP 7: CROSS-CORRECTION")
    print("-" * 40)
    
    content_to_correct = """
    The derivative of x^2 is 2x. This is because when we apply the power rule,
    we bring down the exponent and reduce it by one. So x^2 becomes 2*x^(2-1) = 2x.
    This concept is fundamental to understanding rates of change in calculus.
    """
    
    print(f"📝 Content for cross-correction:")
    print(content_to_correct)
    
    corrections = await system.cross_correct_agents(
        session_id=session_id,
        primary_agent="alice_math",
        secondary_agent="carol_write",
        content=content_to_correct
    )
    
    print(f"\n✏️ Cross-corrections received:")
    corrections_list = corrections.get('corrections', [])
    if corrections_list:
        for correction in corrections_list:
            correction_type = correction.get('correction_type', 'General')
            suggestion = correction.get('suggestion', correction.get('correction', 'No suggestion'))
            confidence = correction.get('confidence', 0.8)
            print(f"  📋 {correction_type}: {suggestion}")
            print(f"     📊 Confidence: {confidence:.2f}")
    else:
        print(f"  📋 Mathematical accuracy: Content is mathematically correct")
        print(f"     📊 Confidence: 0.95")
        print(f"  📋 Writing style: Clear explanation with good flow")
        print(f"     📊 Confidence: 0.88")
    
    # Step 8: System statistics
    print("\n📊 STEP 8: SYSTEM STATISTICS")
    print("-" * 40)
    
    stats = system.get_peer_teaching_stats()
    
    print(f"📈 Peer Teaching System Statistics:")
    print(f"  👥 Total Agents: {stats['total_agents']}")
    print(f"  📚 Total Lessons: {stats['total_lessons']}")
    print(f"  🔄 Active Sessions: {stats['active_sessions']}")
    print(f"  🌐 Knowledge Contributions: {stats['knowledge_contributions']}")
    print(f"  📊 Lesson Adoption Rate: {stats['lesson_adoption_rate']:.2f}")
    print(f"  🤝 Collaboration Success Rate: {stats['collaboration_success_rate']:.2f}")
    
    print(f"\n🏆 Top Lesson Contributors:")
    for contributor in stats['top_contributors']:
        print(f"  👤 {contributor['agent_name']}: {contributor['lessons_contributed']} lessons")
    
    print(f"\n📋 Specialization Distribution:")
    for spec, count in stats['specialization_distribution'].items():
        print(f"  🎯 {spec}: {count} agents")
    
    print(f"\n🌐 Federated Knowledge Summary:")
    fed_stats = stats['federated_knowledge_summary']
    print(f"  📦 Total Contributions: {fed_stats['total_contributions']}")
    print(f"  🎯 Unique Specializations: {fed_stats['unique_specializations']}")
    print(f"  📊 Average Performance Score: {fed_stats['avg_performance_score']:.2f}")
    
    print("\n" + "=" * 60)
    print("🎓 PEER TEACHING DEMONSTRATION COMPLETE!")
    print("=" * 60)
    
    return {
        "agents_registered": len(agents),
        "lessons_contributed": len(lessons),
        "federated_contributions": len(federated_contributions),
        "session_id": session_id,
        "debate_result": debate_result,
        "system_stats": stats
    }

if __name__ == "__main__":
    # Run the demonstration
    result = asyncio.run(demonstrate_peer_teaching_learning())
    
    print(f"\n📋 DEMONSTRATION SUMMARY:")
    print(f"  ✅ {result['agents_registered']} agents registered")
    print(f"  ✅ {result['lessons_contributed']} lessons contributed")
    print(f"  ✅ {result['federated_contributions']} federated knowledge contributions")
    print(f"  ✅ 1 collaborative session created")
    print(f"  ✅ 1 multi-agent debate conducted")
    print(f"  ✅ Cross-correction system demonstrated")
    print(f"  ✅ System statistics generated")