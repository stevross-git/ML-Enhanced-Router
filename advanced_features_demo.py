#!/usr/bin/env python3
"""
Advanced Features Demo
Demonstrates Cross-Persona Memory Inference, Cognitive Loop Debugging,
and Temporal Memory Weighting in the Personal AI Router
"""

import time
import random
from datetime import datetime, timedelta
from cross_persona_memory import get_cross_persona_system
from cognitive_loop_debug import get_cognitive_debugger
from temporal_memory_weighting import get_temporal_memory_system

def main():
    print("🚀 Advanced Features Demo - Personal AI Router")
    print("=" * 60)
    
    # Initialize systems
    print("\n1. Initializing Advanced Systems...")
    cross_persona_system = get_cross_persona_system()
    cognitive_debugger = get_cognitive_debugger()
    temporal_memory_system = get_temporal_memory_system()
    
    # Start a cognitive debugging session
    session_id = cognitive_debugger.start_session("demo_user")
    print(f"✓ Cognitive debugging session started: {session_id}")
    
    # Demo 1: Cross-Persona Memory Inference
    print("\n2. Cross-Persona Memory Inference Demo")
    print("-" * 40)
    
    # Create persona linkages using the analyze_persona_compatibility method
    print("Creating persona linkages...")
    
    # Analyze compatibility between personas
    persona_1_data = {
        "skills": ["analysis", "communication", "problem_solving"],
        "interests": ["technology", "productivity", "efficiency"],
        "preferences": {"formality": 0.8, "structure": 0.9}
    }
    
    persona_2_data = {
        "skills": ["research", "critical_thinking", "problem_solving"],
        "interests": ["science", "technology", "learning"],
        "preferences": {"formality": 0.7, "structure": 0.8}
    }
    
    # Get compatibility analysis
    linkages = cross_persona_system.analyze_persona_compatibility(
        "work", "learning", persona_1_data, persona_2_data
    )
    
    print(f"✓ Analyzed persona compatibility: {len(linkages)} linkages found")
    
    # Get linkage graph
    linkage_graph = cross_persona_system.get_persona_linkage_graph("demo_user")
    print(f"✓ Linkage graph generated with {len(linkage_graph['edges'])} edges")
    
    # Generate cross-persona insights
    persona_data = {
        "work": {
            "skills": ["data_analysis", "project_management", "communication"],
            "interests": ["efficiency", "innovation", "problem_solving"],
            "traits": ["analytical", "methodical", "results_oriented"]
        },
        "creative": {
            "skills": ["writing", "design", "storytelling"],
            "interests": ["art", "literature", "music"],
            "traits": ["imaginative", "expressive", "intuitive"]
        },
        "learning": {
            "skills": ["research", "synthesis", "critical_thinking"],
            "interests": ["science", "technology", "philosophy"],
            "traits": ["curious", "persistent", "open_minded"]
        }
    }
    
    insights = cross_persona_system.generate_cross_persona_insights("demo_user", persona_data)
    print(f"✓ Generated {len(insights)} cross-persona insights")
    
    for insight in insights[:2]:  # Show first 2 insights
        print(f"  • {insight.insight_type}: {insight.description}")
        print(f"    Confidence: {insight.confidence:.2f}")
        print(f"    Suggestion: {insight.suggestion}")
    
    # Demo 2: Cognitive Loop Debugging
    print("\n3. Cognitive Loop Debugging Demo")
    print("-" * 40)
    
    # Log various decision types
    print("Logging AI decision-making processes...")
    
    # Persona switch decision
    cognitive_debugger.log_persona_switch(
        from_persona="work",
        to_persona="creative",
        reason="User query involves creative problem-solving",
        context={"query": "Help me brainstorm innovative solutions for our project"},
        influencing_memories=["Previous creative successes", "User preference for innovation"],
        confidence=0.8,
        user_id="demo_user"
    )
    
    # Tone adjustment decision
    cognitive_debugger.log_tone_adjustment(
        original_tone="formal",
        adjusted_tone="encouraging",
        mood_detected="frustrated",
        confidence=0.75,
        context={"query": "I'm having trouble with this task"},
        user_id="demo_user"
    )
    
    # Memory retrieval decision
    cognitive_debugger.log_memory_retrieval(
        query="What are my productivity preferences?",
        retrieved_memories=[
            {"memory": "Prefers morning work sessions", "relevance": 0.9},
            {"memory": "Uses time-blocking method", "relevance": 0.8}
        ],
        selection_criteria="Relevance to productivity and time management",
        confidence=0.85,
        context={"persona": "work", "time_of_day": "morning"},
        user_id="demo_user"
    )
    
    # Get decision patterns
    patterns = cognitive_debugger.get_decision_patterns("demo_user")
    print(f"✓ Analyzed {patterns.get('total_decisions', 0)} decisions")
    print(f"  Most common decision type: {patterns.get('most_common_type', 'N/A')}")
    print(f"  Average confidence: {patterns.get('average_confidence', 0):.2f}")
    
    # Demo 3: Temporal Memory Weighting
    print("\n4. Temporal Memory Weighting Demo")
    print("-" * 40)
    
    # Register memories with different priorities
    print("Registering memories with temporal weighting...")
    
    memories = [
        ("important_meeting", 0.9, "high", "Meeting with CEO about project direction"),
        ("lunch_preference", 0.6, "medium", "User prefers Italian food"),
        ("casual_comment", 0.3, "low", "Random comment about weather"),
        ("critical_password", 0.95, "critical", "Security credentials for system"),
        ("daily_routine", 0.7, "medium", "User's morning routine preferences")
    ]
    
    for memory_id, confidence, priority, description in memories:
        # Create memory with different timestamps to simulate age
        created_at = datetime.now() - timedelta(days=random.randint(1, 30))
        
        # Import the enum directly
        from temporal_memory_weighting import MemoryPriority
        
        # Convert string to enum
        priority_map = {
            "low": MemoryPriority.LOW,
            "medium": MemoryPriority.MEDIUM,
            "high": MemoryPriority.HIGH,
            "critical": MemoryPriority.CRITICAL
        }
        
        temporal_memory_system.register_memory(
            memory_id=memory_id,
            initial_confidence=confidence,
            priority=priority_map[priority],
            created_at=created_at
        )
        
        print(f"  ✓ Registered: {memory_id} (confidence: {confidence:.2f}, priority: {priority})")
    
    # Simulate memory access patterns
    print("\nSimulating memory access patterns...")
    
    # Frequently access important memories
    for _ in range(5):
        temporal_memory_system.access_memory("important_meeting")
        temporal_memory_system.access_memory("critical_password")
    
    # Occasionally access medium priority memories
    for _ in range(2):
        temporal_memory_system.access_memory("lunch_preference")
        temporal_memory_system.access_memory("daily_routine")
    
    # Never access low priority memory (let it decay)
    # temporal_memory_system.access_memory("casual_comment")  # Commented out intentionally
    
    print("✓ Memory access patterns simulated")
    
    # Get memory insights
    print("\nMemory insights:")
    for memory_id, _, _, _ in memories:
        insights = temporal_memory_system.get_memory_insights(memory_id)
        
        if "error" not in insights:
            current_conf = insights["temporal_status"]["current_confidence"]
            access_count = insights["access_patterns"]["access_count"]
            print(f"  • {memory_id}: confidence={current_conf:.2f}, accessed={access_count} times")
    
    # Get cleanup suggestions
    suggestions = temporal_memory_system.suggest_memory_cleanup("demo_user")
    if suggestions:
        print(f"\n💡 Memory cleanup suggestions: {len(suggestions)} memories need attention")
        for suggestion in suggestions[:2]:
            print(f"  • {suggestion['memory_id']}: {suggestion['reason']}")
    
    # Demo 4: Integration Example
    print("\n5. Advanced Features Integration Demo")
    print("-" * 40)
    
    # Show how systems work together
    print("Demonstrating system integration...")
    
    # Get system statistics
    cross_persona_stats = cross_persona_system.get_statistics()
    cognitive_stats = cognitive_debugger.get_statistics()
    temporal_stats = temporal_memory_system.get_statistics()
    
    print(f"✓ Cross-persona system: {cross_persona_stats.get('total_linkages', 0)} linkages")
    print(f"✓ Cognitive debugger: {cognitive_stats.get('total_decisions', 0)} decisions logged")
    print(f"✓ Temporal memory: {temporal_stats.get('total_memories', 0)} memories tracked")
    
    # Show decision explanation
    decisions = cognitive_debugger.get_user_decisions("demo_user", limit=1)
    if decisions:
        decision = decisions[0]
        explanation = cognitive_debugger.explain_decision(decision.decision_id)
        print(f"\n🔍 Decision explanation for {decision.decision_type.value}:")
        
        # Handle different explanation structures
        if isinstance(explanation, dict):
            if 'decision' in explanation:
                print(f"  Decision: {explanation['decision'].get('decision_made', 'N/A')}")
                print(f"  Reasoning: {explanation['decision'].get('reasoning', 'N/A')}")
                print(f"  Confidence: {explanation['decision'].get('confidence', 0):.2f}")
            else:
                print(f"  Decision: {decision.decision_made}")
                print(f"  Reasoning: {decision.reasoning}")
                print(f"  Confidence: {decision.confidence:.2f}")
        else:
            print(f"  Decision: {decision.decision_made}")
            print(f"  Reasoning: {decision.reasoning}")
            print(f"  Confidence: {decision.confidence:.2f}")
    
    # End session
    cognitive_debugger.end_session(session_id)
    print(f"\n✓ Cognitive debugging session ended: {session_id}")
    
    print("\n🎉 Advanced Features Demo Complete!")
    print("=" * 60)
    print("The Personal AI Router now includes:")
    print("• Cross-Persona Memory Inference for knowledge bridging")
    print("• Cognitive Loop Debugging for AI transparency")
    print("• Temporal Memory Weighting for time-aware relevance")
    print("• Full integration with existing Personal AI features")

if __name__ == "__main__":
    main()