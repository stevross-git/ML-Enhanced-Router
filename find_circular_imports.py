#!/usr/bin/env python3
"""
Find circular imports in your ML Router project
Run this to identify which modules are causing the circular dependency
"""

import os
import sys
import ast
import re
from pathlib import Path

def find_imports_in_file(filepath):
    """Extract all imports from a Python file"""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
        
        tree = ast.parse(content)
        imports = []
        
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                if node.module:
                    imports.append(node.module)
        
        return imports
    except Exception as e:
        print(f"Error reading {filepath}: {e}")
        return []

def build_dependency_graph():
    """Build a dependency graph of all Python files"""
    current_dir = Path('.')
    python_files = list(current_dir.glob('*.py'))
    
    # Also check common subdirectories
    for subdir in ['src', 'lib', 'modules', 'components']:
        if Path(subdir).exists():
            python_files.extend(Path(subdir).glob('*.py'))
    
    dependency_graph = {}
    
    for py_file in python_files:
        if py_file.name.startswith('.'):
            continue
            
        module_name = py_file.stem
        imports = find_imports_in_file(py_file)
        
        # Filter to only local imports (files that exist in project)
        local_imports = []
        for imp in imports:
            # Check if this import corresponds to a local file
            if imp in [f.stem for f in python_files]:
                local_imports.append(imp)
        
        dependency_graph[module_name] = local_imports
        print(f"📁 {module_name}: imports {local_imports}")
    
    return dependency_graph

def find_cycles(graph):
    """Find circular dependencies using DFS"""
    def dfs(node, path, visited, rec_stack):
        if node in rec_stack:
            # Found a cycle
            cycle_start = rec_stack.index(node)
            cycle = rec_stack[cycle_start:] + [node]
            return cycle
        
        if node in visited:
            return None
        
        visited.add(node)
        rec_stack.append(node)
        
        for neighbor in graph.get(node, []):
            cycle = dfs(neighbor, path + [neighbor], visited, rec_stack)
            if cycle:
                return cycle
        
        rec_stack.pop()
        return None
    
    cycles = []
    visited = set()
    
    for node in graph:
        if node not in visited:
            cycle = dfs(node, [node], visited, [])
            if cycle:
                cycles.append(cycle)
    
    return cycles

def check_specific_problematic_imports():
    """Check for known problematic import patterns"""
    print("\n🔍 CHECKING FOR KNOWN PROBLEMATIC PATTERNS:")
    print("=" * 50)
    
    problematic_patterns = [
        ('app.py imports models', r'from models import'),
        ('models.py imports app', r'from app import'),
        ('models.py imports db from app', r'from app import.*db'),
        ('app.py imports router components early', r'from ml_router import'),
        ('circular SQLAlchemy imports', r'from.*db.*import'),
        ('Flask app context issues', r'app\.app_context'),
    ]
    
    python_files = list(Path('.').glob('*.py'))
    
    for pattern_name, pattern in problematic_patterns:
        print(f"\n🔍 Checking: {pattern_name}")
        found_issues = []
        
        for py_file in python_files:
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                matches = re.findall(pattern, content, re.IGNORECASE)
                if matches:
                    found_issues.append(f"  📄 {py_file.name}: {matches}")
            except Exception as e:
                continue
        
        if found_issues:
            print(f"  ❌ FOUND ISSUES:")
            for issue in found_issues:
                print(issue)
        else:
            print(f"  ✅ No issues found")

def analyze_specific_files():
    """Analyze the most likely problematic files"""
    print("\n🔍 ANALYZING CRITICAL FILES:")
    print("=" * 50)
    
    critical_files = [
        'app.py',
        'models.py', 
        'ml_router.py',
        'config.py',
        'model_manager.py',
        'ai_models.py'
    ]
    
    for filename in critical_files:
        if Path(filename).exists():
            print(f"\n📄 ANALYZING {filename}:")
            imports = find_imports_in_file(filename)
            
            # Check for problematic imports
            problematic = []
            for imp in imports:
                if imp in critical_files:
                    problematic.append(imp)
            
            if problematic:
                print(f"  ❌ IMPORTS OTHER CRITICAL FILES: {problematic}")
            else:
                print(f"  ✅ No circular imports with critical files")
            
            print(f"  📋 All imports: {imports[:10]}{'...' if len(imports) > 10 else ''}")
        else:
            print(f"  ⚠️ {filename} not found")

def main():
    print("🔍 CIRCULAR IMPORT DETECTOR")
    print("=" * 50)
    print("Analyzing your ML Router project for circular dependencies...\n")
    
    # Build dependency graph
    print("📊 BUILDING DEPENDENCY GRAPH:")
    dependency_graph = build_dependency_graph()
    
    # Find cycles
    print(f"\n🔄 SEARCHING FOR CIRCULAR DEPENDENCIES:")
    cycles = find_cycles(dependency_graph)
    
    if cycles:
        print(f"❌ FOUND {len(cycles)} CIRCULAR DEPENDENCIES:")
        for i, cycle in enumerate(cycles, 1):
            print(f"\n🔄 Cycle #{i}:")
            cycle_str = " → ".join(cycle)
            print(f"   {cycle_str}")
    else:
        print("✅ No obvious circular dependencies found in dependency graph")
    
    # Check specific patterns
    check_specific_problematic_imports()
    
    # Analyze critical files
    analyze_specific_files()
    
    print(f"\n💡 RECOMMENDATIONS:")
    print("=" * 50)
    print("1. If circular imports found: Break the cycle by:")
    print("   - Moving shared code to a separate module")
    print("   - Using delayed imports (import inside functions)")
    print("   - Restructuring the dependency chain")
    print("\n2. Common fixes for Flask + SQLAlchemy:")
    print("   - Move model definitions to separate files")
    print("   - Import db object consistently")
    print("   - Use application factory pattern")
    print("\n3. Check the most likely culprits:")
    print("   - app.py importing from models.py")
    print("   - models.py importing from app.py") 
    print("   - Multiple files importing the same database instance")

if __name__ == '__main__':
    main()
