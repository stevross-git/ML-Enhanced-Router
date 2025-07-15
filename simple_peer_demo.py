#!/usr/bin/env python3
"""
Simple Peer Teaching Demo
Shows how the learning system works step by step
"""

import asyncio
from peer_teaching_system import (
    PeerTeachingSystem, 
    AgentSpecialization, 
    LessonType, 
    get_peer_teaching_system
)

async def simple_learning_demo():
    """Simple demonstration of peer teaching system learning"""
    print("🎓 PEER TEACHING LEARNING DEMO")
    print("=" * 50)
    
    system = get_peer_teaching_system()
    
    # Step 1: Register agents
    print("\n1️⃣ REGISTERING AGENTS")
    print("-" * 30)
    
    # Math expert
    system.register_agent("alice", "Alice", AgentSpecialization.MATH_SOLVER, ["algebra", "calculus"])
    print("✅ Alice registered as Math Solver")
    
    # Writing expert  
    system.register_agent("bob", "Bob", AgentSpecialization.WRITING_ASSISTANT, ["grammar", "style"])
    print("✅ Bob registered as Writing Assistant")
    
    # Research expert
    system.register_agent("carol", "Carol", AgentSpecialization.RESEARCH_ASSISTANT, ["analysis", "facts"])
    print("✅ Carol registered as Research Assistant")
    
    # Step 2: Contribute lessons
    print("\n2️⃣ CONTRIBUTING LESSONS")
    print("-" * 30)
    
    # Alice contributes a math lesson
    lesson1_id = system.contribute_lesson(
        agent_id="alice",
        lesson_type=LessonType.STRATEGY,
        domain="mathematics",
        title="Solving Quadratic Equations",
        content="Use the quadratic formula when factoring fails. Check discriminant first.",
        strategy_steps=["Standard form", "Calculate discriminant", "Apply formula", "Check solutions"],
        effectiveness_score=0.95,
        usage_context="algebra problems",
        success_metrics={"accuracy": 0.95, "speed": 0.8}
    )
    print("📚 Alice contributed: 'Solving Quadratic Equations'")
    
    # Bob contributes a writing lesson
    lesson2_id = system.contribute_lesson(
        agent_id="bob",
        lesson_type=LessonType.PATTERN,
        domain="writing",
        title="Active Voice Conversion",
        content="Convert passive to active voice for clearer writing. Find the doer and make them the subject.",
        strategy_steps=["Identify passive voice", "Find the doer", "Make doer the subject", "Use active verbs"],
        effectiveness_score=0.88,
        usage_context="essay writing",
        success_metrics={"clarity": 0.92, "engagement": 0.85}
    )
    print("📚 Bob contributed: 'Active Voice Conversion'")
    
    # Step 3: Discover and adopt lessons
    print("\n3️⃣ DISCOVERING LESSONS")
    print("-" * 30)
    
    # Carol (research assistant) looks for math lessons
    math_lessons = system.find_relevant_lessons("carol", "mathematics")
    print(f"🔍 Carol found {len(math_lessons)} math lessons")
    
    if math_lessons:
        lesson = math_lessons[0]
        print(f"📖 Found: '{lesson.title}' (effectiveness: {lesson.effectiveness_score:.2f})")
        
        # Carol adopts the lesson
        if system.adopt_lesson("carol", lesson.lesson_id):
            print("✅ Carol adopted Alice's math lesson!")
    
    # Bob looks for research lessons (none exist yet)
    research_lessons = system.find_relevant_lessons("bob", "research")
    print(f"🔍 Bob found {len(research_lessons)} research lessons")
    
    # Step 4: Federated knowledge sharing
    print("\n4️⃣ FEDERATED KNOWLEDGE SHARING")
    print("-" * 30)
    
    # Alice shares anonymized knowledge
    contrib1_id = system.contribute_federated_knowledge(
        agent_id="alice",
        specialization=AgentSpecialization.MATH_SOLVER,
        query_type="quadratic_equations",
        toolkit_used="algebraic_methods",
        approach_summary="Factor-first approach with quadratic formula backup",
        performance_metrics={"accuracy": 0.95, "speed": 0.80},
        lessons_learned=["Check discriminant first", "Factoring is faster"],
        optimization_tips=["Memorize common factors", "Use completing the square for insights"],
        error_patterns=["Sign errors", "Arithmetic mistakes in discriminant"]
    )
    print("🌐 Alice shared federated knowledge about quadratic equations")
    
    # Retrieve federated knowledge
    fed_knowledge = system.get_federated_knowledge(AgentSpecialization.MATH_SOLVER, "quadratic_equations")
    print(f"📊 Retrieved {len(fed_knowledge)} federated knowledge entries")
    
    if fed_knowledge:
        knowledge = fed_knowledge[0]
        print(f"   📋 Approach: {knowledge.approach_summary}")
        print(f"   📈 Performance: {knowledge.performance_metrics}")
    
    # Step 5: Multi-agent collaboration
    print("\n5️⃣ COLLABORATIVE SESSION")
    print("-" * 30)
    
    # Start collaborative session
    session_id = await system.start_collaborative_session(
        initiator_agent="alice",
        task_description="Create a math tutoring guide",
        session_type="educational_content",
        required_specializations=[AgentSpecialization.MATH_SOLVER, AgentSpecialization.WRITING_ASSISTANT]
    )
    print(f"🤝 Started collaborative session: {session_id[:8]}...")
    
    # Multi-agent debate
    debate_result = await system.multi_agent_debate(
        session_id=session_id,
        question="What's the best way to teach quadratic equations?"
    )
    print("🗣️ Conducted multi-agent debate")
    print(f"   🎯 Agents participated and reached consensus")
    
    # Step 6: System statistics
    print("\n6️⃣ SYSTEM STATISTICS")
    print("-" * 30)
    
    stats = system.get_peer_teaching_stats()
    print(f"👥 Total Agents: {stats['total_agents']}")
    print(f"📚 Total Lessons: {stats['total_lessons']}")
    print(f"🔄 Active Sessions: {stats['active_sessions']}")
    print(f"🌐 Knowledge Contributions: {stats['knowledge_contributions']}")
    print(f"📊 Lesson Adoption Rate: {stats['lesson_adoption_rate']:.2f}")
    print(f"🤝 Collaboration Success Rate: {stats['collaboration_success_rate']:.2f}")
    
    print("\n" + "=" * 50)
    print("🎯 LEARNING DEMO COMPLETE!")
    print("=" * 50)
    
    print(f"\n📋 KEY LEARNING FEATURES DEMONSTRATED:")
    print(f"✅ Agent registration with specializations")
    print(f"✅ Lesson contribution and knowledge sharing")
    print(f"✅ Cross-domain lesson discovery and adoption")
    print(f"✅ Federated learning with anonymized knowledge")
    print(f"✅ Multi-agent collaboration and debate")
    print(f"✅ System-wide statistics and tracking")
    
    return {
        "agents": 3,
        "lessons": 2,
        "federated_contributions": 1,
        "collaborations": 1,
        "session_id": session_id
    }

if __name__ == "__main__":
    result = asyncio.run(simple_learning_demo())
    print(f"\n🎉 Demo completed with {result['agents']} agents and {result['lessons']} lessons!")