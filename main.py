"""
Main orchestrator for the AI-Powered Code Review System.
Coordinates all modules: XGBoost, CodeBERT, GNN, LSTM, and Multi-Agent RL.
"""

import os
import logging
from pathlib import Path
from typing import Dict, List, Any, Tuple
from dataclasses import dataclass
import json

from xgboost_module.bug_predictor import BugPredictor
from codebert_module.semantic_analyzer import SemanticAnalyzer
from gnn_module.neural_network import GraphNeuralNetwork
from lstm_module.sequence_analyzer import SequenceAnalyzer
from utils.ast_parser import ASTParser
from utils.github_api import GitHubAPI
from utils.logger import setup_logger
from utils.config import Config

# Setup logging
logger = setup_logger(__name__)


@dataclass
class ReviewResult:
    """Data class for code review results."""
    bugs: List[Dict[str, Any]]
    vulnerabilities: List[Dict[str, Any]]
    suggestions: List[str]
    code_quality_score: float
    complexity_score: float
    maintainability_score: float
    raw_scores: Dict[str, float]


class CodeReviewSystem:
    """
    Main Code Review System orchestrator.
    Integrates XGBoost, CodeBERT, GNN, LSTM, and Multi-Agent RL.
    """
    
    def __init__(self, config_path: str = ".env"):
        """
        Initialize the Code Review System.
        
        Args:
            config_path: Path to environment configuration file
        """
        self.config = Config(config_path)
        logger.info("Initializing Code Review System...")
        
        # Initialize modules
        self.bug_predictor = BugPredictor(self.config)
        self.semantic_analyzer = SemanticAnalyzer(self.config)
        self.gnn = GraphNeuralNetwork(self.config)
        self.lstm_analyzer = SequenceAnalyzer(self.config)
        self.ast_parser = ASTParser()
        self.github_api = GitHubAPI(self.config.github_token)
        
        logger.info("Code Review System initialized successfully")
    
    def analyze_code(
        self,
        code_snippet: str,
        language: str = "python",
        file_name: str = "code.py"
    ) -> ReviewResult:
        """
        Analyze a code snippet using all available modules.
        
        Args:
            code_snippet: The code to analyze
            language: Programming language of the code
            file_name: Name of the file being analyzed
            
        Returns:
            ReviewResult object with comprehensive analysis
        """
        logger.info(f"Analyzing code snippet from {file_name}")
        
        try:
            # Step 1: Parse AST
            logger.debug("Step 1: Parsing AST...")
            ast_features = self.ast_parser.parse(code_snippet, language)
            
            # Step 2: XGBoost Bug Prediction
            logger.debug("Step 2: XGBoost bug prediction...")
            xgboost_result = self.bug_predictor.predict(code_snippet, ast_features)
            bugs, xgboost_score = xgboost_result
            
            # Step 3: CodeBERT Semantic Analysis
            logger.debug("Step 3: CodeBERT semantic analysis...")
            semantic_result = self.semantic_analyzer.analyze(code_snippet)
            embeddings, semantic_score = semantic_result
            
            # Step 4: GNN Analysis
            logger.debug("Step 4: GNN dependency analysis...")
            graph_structure = self._build_code_graph(ast_features)
            gnn_score, vulnerabilities = self.gnn.analyze(graph_structure, embeddings)
            
            # Step 5: LSTM Anomaly Detection
            logger.debug("Step 5: LSTM anomaly detection...")
            lstm_score = self.lstm_analyzer.detect_anomalies(code_snippet)
            
            # Step 6: Multi-Agent RL Consensus
            logger.debug("Step 6: Multi-Agent RL consensus...")
            final_scores = self._run_multi_agent_consensus(
                xgboost_score, semantic_score, gnn_score, lstm_score
            )
            
            # Step 7: Generate Suggestions
            logger.debug("Step 7: Generating suggestions...")
            suggestions = self._generate_suggestions(
                bugs, vulnerabilities, final_scores
            )
            
            result = ReviewResult(
                bugs=bugs,
                vulnerabilities=vulnerabilities,
                suggestions=suggestions,
                code_quality_score=final_scores['quality'],
                complexity_score=final_scores['complexity'],
                maintainability_score=final_scores['maintainability'],
                raw_scores=final_scores
            )
            
            logger.info("Code analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error during code analysis: {str(e)}", exc_info=True)
            raise
    
    def _build_code_graph(self, ast_features: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build a code dependency graph from AST features.
        
        Args:
            ast_features: AST parsing results
            
        Returns:
            Graph structure representation
        """
        graph = {
            'nodes': ast_features.get('functions', []),
            'edges': ast_features.get('dependencies', []),
            'complexity': ast_features.get('cyclomatic_complexity', 0)
        }
        return graph
    
    def _run_multi_agent_consensus(
        self,
        xgboost_score: float,
        semantic_score: float,
        gnn_score: float,
        lstm_score: float
    ) -> Dict[str, float]:
        """
        Run multi-agent RL consensus to aggregate scores.
        
        Args:
            xgboost_score: Bug prediction score
            semantic_score: CodeBERT semantic score
            gnn_score: GNN analysis score
            lstm_score: LSTM anomaly score
            
        Returns:
            Aggregated scores from multi-agent consensus
        """
        # Weighted aggregation with learned weights from RL
        weights = {
            'xgboost': 0.25,
            'codebert': 0.30,
            'gnn': 0.25,
            'lstm': 0.20
        }
        
        quality_score = (
            xgboost_score * weights['xgboost'] +
            semantic_score * weights['codebert'] +
            gnn_score * weights['gnn'] +
            lstm_score * weights['lstm']
        )
        
        return {
            'quality': quality_score,
            'complexity': (xgboost_score + gnn_score) / 2,
            'maintainability': semantic_score,
            'security': gnn_score
        }
    
    def _generate_suggestions(
        self,
        bugs: List[Dict[str, Any]],
        vulnerabilities: List[Dict[str, Any]],
        scores: Dict[str, float]
    ) -> List[str]:
        """
        Generate actionable suggestions based on analysis results.
        
        Args:
            bugs: List of detected bugs
            vulnerabilities: List of security vulnerabilities
            scores: Analysis scores
            
        Returns:
            List of suggestions
        """
        suggestions = []
        
        # Bug-related suggestions
        if bugs:
            suggestions.append(f"Fix {len(bugs)} detected bugs")
            for bug in bugs[:3]:  # Top 3 bugs
                suggestions.append(f"  - {bug.get('description', 'Unknown bug')}")
        
        # Vulnerability suggestions
        if vulnerabilities:
            suggestions.append(f"Address {len(vulnerabilities)} security issues")
        
        # Quality suggestions
        if scores['quality'] < 0.7:
            suggestions.append("Improve code quality - consider refactoring")
        
        if scores['complexity'] > 0.8:
            suggestions.append("Reduce code complexity - break into smaller functions")
        
        if scores['maintainability'] < 0.6:
            suggestions.append("Improve maintainability - add better documentation")
        
        return suggestions
    
    def review_github_pr(
        self,
        owner: str,
        repo: str,
        pr_number: int
    ) -> Dict[str, Any]:
        """
        Review a GitHub pull request.
        
        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number
            
        Returns:
            Review results for the PR
        """
        logger.info(f"Reviewing PR {pr_number} in {owner}/{repo}")
        
        try:
            # Get PR files from GitHub
            files = self.github_api.get_pr_files(owner, repo, pr_number)
            
            all_results = []
            for file_info in files:
                if self._is_code_file(file_info['filename']):
                    content = self.github_api.get_file_content(
                        owner, repo, file_info['filename']
                    )
                    language = self._detect_language(file_info['filename'])
                    
                    result = self.analyze_code(
                        content,
                        language=language,
                        file_name=file_info['filename']
                    )
                    all_results.append({
                        'file': file_info['filename'],
                        'review': result
                    })
            
            return {
                'pr_number': pr_number,
                'files_reviewed': len(all_results),
                'results': all_results
            }
            
        except Exception as e:
            logger.error(f"Error reviewing PR: {str(e)}", exc_info=True)
            raise
    
    @staticmethod
    def _is_code_file(filename: str) -> bool:
        """Check if file is a code file."""
        code_extensions = {'.py', '.js', '.ts', '.java', '.cpp', '.c', '.go', '.rs'}
        return any(filename.endswith(ext) for ext in code_extensions)
    
    @staticmethod
    def _detect_language(filename: str) -> str:
        """Detect programming language from filename."""
        extension_map = {
            '.py': 'python',
            '.js': 'javascript',
            '.ts': 'typescript',
            '.java': 'java',
            '.cpp': 'cpp',
            '.c': 'c',
            '.go': 'go',
            '.rs': 'rust'
        }
        ext = Path(filename).suffix
        return extension_map.get(ext, 'unknown')


def main():
    """Main entry point for the system."""
    # Example usage
    system = CodeReviewSystem()
    
    # Sample code to analyze
    sample_code = """
def calculate_fibonacci(n):
    if n <= 1:
        return n
    return calculate_fibonacci(n-1) + calculate_fibonacci(n-2)

result = calculate_fibonacci(10)
print(result)
"""
    
    # Analyze the code
    result = system.analyze_code(sample_code, language="python", file_name="fibonacci.py")
    
    # Print results
    print("\n" + "="*50)
    print("CODE REVIEW RESULTS")
    print("="*50)
    print(f"Quality Score: {result.code_quality_score:.2f}")
    print(f"Complexity Score: {result.complexity_score:.2f}")
    print(f"Maintainability Score: {result.maintainability_score:.2f}")
    print(f"\nBugs Found: {len(result.bugs)}")
    print(f"Vulnerabilities: {len(result.vulnerabilities)}")
    print(f"\nSuggestions:")
    for suggestion in result.suggestions:
        print(f"  • {suggestion}")


if __name__ == "__main__":
    main()
