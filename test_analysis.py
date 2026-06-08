from main import CodeReviewSystem
import sys

system = CodeReviewSystem()

test_cases = {
    "SIMPLE_FUNCTION": """
def greet(name):
    return f"Hello, {name}!"

print(greet("Alice"))
""",

    "COMPLEX_FUNCTION": """
def calculate(a, b, c):
    if a > 0:
        if b > 0:
            if c > 0:
                if a + b > c:
                    if a + c > b:
                        if b + c > a:
                            return a + b + c
    return 0
""",

    "BUGGY_CODE": """
import unused_module
import os

def process():
    TODO: implement this
    x = 1
    y = 2
    if x > 0:
        if y > 0:
            if x + y > 0:
                return x + y
    return None
""",

    "VULNERABLE_CODE": """
def execute_command(user_input):
    os.system(user_input)
    
def eval_code(code):
    eval(code)
"""
}

print("\n" + "="*70)
print("🔍 INTERACTIVE CODE REVIEW SYSTEM")
print("="*70)

for test_name, code in test_cases.items():
    print(f"\n📝 Analyzing: {test_name}")
    print("-" * 70)
    
    result = system.analyze_code(code, language="python")
    
    # Display results in a nice format
    print(f"\n📊 QUALITY METRICS:")
    print(f"   Quality Score:      {result.code_quality_score:.2%}")
    print(f"   Complexity:         {result.complexity_score:.2%}")
    print(f"   Maintainability:    {result.maintainability_score:.2%}")
    
    # Color coded status
    if result.code_quality_score > 0.75:
        status = "✅ EXCELLENT"
    elif result.code_quality_score > 0.6:
        status = "⚠️  ACCEPTABLE"
    else:
        status = "❌ NEEDS WORK"
    
    print(f"   Overall Status:     {status}")
    
    # Bugs
    if result.bugs:
        print(f"\n🐛 BUGS DETECTED: {len(result.bugs)}")
        for i, bug in enumerate(result.bugs, 1):
            severity_emoji = "🔴" if bug['severity'] == 'high' else "🟡" if bug['severity'] == 'medium' else "🟢"
            print(f"   {i}. {severity_emoji} [{bug['severity'].upper()}] {bug['type']}")
            print(f"      └─ {bug['description']}")
    else:
        print(f"\n✅ NO BUGS DETECTED")
    
    # Vulnerabilities
    if result.vulnerabilities:
        print(f"\n⚠️  VULNERABILITIES: {len(result.vulnerabilities)}")
        for i, vuln in enumerate(result.vulnerabilities, 1):
            print(f"   {i}. 🔒 [{vuln['severity'].upper()}] {vuln['type']}")
            print(f"      └─ {vuln['description']}")
    
    # Suggestions
    if result.suggestions:
        print(f"\n💡 SUGGESTIONS:")
        for i, suggestion in enumerate(result.suggestions, 1):
            print(f"   {i}. {suggestion}")
    
    print()

print("="*70)
print("✅ Analysis Complete!")
print("="*70)