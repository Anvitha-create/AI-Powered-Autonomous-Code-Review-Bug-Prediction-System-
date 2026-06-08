"""AST parsing for code analysis."""

import ast
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


class ASTParser:
    """Parse and analyze Abstract Syntax Trees."""
    
    def parse(self, code: str, language: str = 'python') -> Dict[str, Any]:
        """
        Parse code and extract AST features.
        
        Args:
            code: Source code to parse
            language: Programming language
            
        Returns:
            Dictionary of AST features
        """
        if language == 'python':
            return self._parse_python(code)
        else:
            logger.warning(f"Language {language} not fully supported, returning basic features")
            return self._parse_generic(code)
    
    def _parse_python(self, code: str) -> Dict[str, Any]:
        """Parse Python code."""
        try:
            tree = ast.parse(code)
            
            functions = []
            classes = []
            imports = []
            dependencies = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        'name': node.name,
                        'lineno': node.lineno,
                        'args': len(node.args.args)
                    })
                elif isinstance(node, ast.ClassDef):
                    classes.append({
                        'name': node.name,
                        'lineno': node.lineno
                    })
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    try:
                        if hasattr(ast, 'unparse'):
                            imports.append(ast.unparse(node))
                        else:
                            imports.append(str(node))
                    except:
                        pass
            
            complexity = self._calculate_complexity(tree)
            nesting = self._calculate_max_nesting(tree)
            
            return {
                'functions': functions,
                'classes': classes,
                'imports': imports,
                'dependencies': dependencies,
                'cyclomatic_complexity': complexity,
                'max_nesting_depth': nesting,
                'unused_imports': self._find_unused_imports(code, imports),
                'complexity_line': 0
            }
        except SyntaxError as e:
            logger.error(f"Syntax error parsing Python: {e}")
            return self._parse_generic(code)
    
    def _parse_generic(self, code: str) -> Dict[str, Any]:
        """Parse code generically when language is not supported."""
        lines = code.split('\n')
        
        return {
            'functions': [],
            'classes': [],
            'imports': [],
            'dependencies': [],
            'cyclomatic_complexity': code.count('if') + code.count('for') + code.count('while'),
            'max_nesting_depth': self._estimate_nesting(code),
            'unused_imports': [],
            'complexity_line': 0,
            'line_count': len(lines)
        }
    
    def _calculate_complexity(self, tree: ast.AST) -> int:
        """Calculate cyclomatic complexity."""
        complexity = 1
        for node in ast.walk(tree):
            if isinstance(node, (ast.If, ast.While, ast.For, ast.ExceptHandler)):
                complexity += 1
        return complexity
    
    def _calculate_max_nesting(self, tree: ast.AST, depth: int = 0) -> int:
        """Calculate maximum nesting depth."""
        max_depth = depth
        for child in ast.iter_child_nodes(tree):
            child_depth = self._calculate_max_nesting(child, depth + 1)
            max_depth = max(max_depth, child_depth)
        return max_depth
    
    def _find_unused_imports(self, code: str, imports: List[str]) -> List[str]:
        """Find unused imports."""
        unused = []
        for imp in imports:
            # Simple heuristic - check if import name appears in code
            import_name = imp.split()[-1] if 'import' in imp else imp
            if import_name not in code:
                unused.append(imp)
        return unused
    
    def _estimate_nesting(self, code: str) -> int:
        """Estimate nesting depth."""
        max_indent = 0
        for line in code.split('\n'):
            if line.strip():
                indent = len(line) - len(line.lstrip())
                max_indent = max(max_indent, indent)
        return max_indent // 4
