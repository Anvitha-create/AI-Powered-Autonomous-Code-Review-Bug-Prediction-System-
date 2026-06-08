"""XGBoost Module for bug prediction and code quality scoring."""

import numpy as np
import xgboost as xgb
from typing import Tuple, List, Dict, Any
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class BugPredictor:
    """XGBoost-based bug prediction and classification."""
    
    def __init__(self, config):
        """
        Initialize the bug predictor.
        
        Args:
            config: Configuration object
        """
        self.config = config
        self.model = None
        self.feature_names = self._get_feature_names()
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize or load the XGBoost model."""
        model_path = self.config.xgboost_model_path
        
        if Path(model_path).exists():
            logger.info(f"Loading XGBoost model from {model_path}")
            try:
                self.model = xgb.Booster()
                self.model.load_model(model_path)
            except Exception as e:
                logger.warning(f"Could not load model: {e}, creating new model")
                self.model = None
        
        if self.model is None:
            logger.info("Creating new XGBoost model")
            try:
                self.model = xgb.XGBClassifier(
                    max_depth=7,
                    learning_rate=0.1,
                    n_estimators=200,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective='binary:logistic',
                    random_state=42,
                    verbosity=0
                )
            except Exception as e:
                logger.error(f"Error creating XGBoost model: {e}")
                self.model = None
    
    def predict(
        self,
        code_snippet: str,
        ast_features: Dict[str, Any]
    ) -> Tuple[List[Dict[str, Any]], float]:
        """
        Predict bugs in code snippet.
        
        Args:
            code_snippet: Code to analyze
            ast_features: AST parsing results
            
        Returns:
            Tuple of (bugs list, quality score)
        """
        features = self._extract_features(code_snippet, ast_features)
        
        if self.model is None:
            return [], 0.5
        
        try:
            X = np.array([features])
            
            # For untrained model, just use heuristics
            if not hasattr(self.model, 'n_features_in_'):
                bugs = self._identify_bugs(features, 0.5, ast_features)
                return bugs, 0.5
            
            predictions = self.model.predict(X)
            probabilities = self.model.predict_proba(X)
            
            bug_probability = float(probabilities[0][1])
            quality_score = 1.0 - bug_probability
            
            bugs = self._identify_bugs(features, bug_probability, ast_features)
            
            return bugs, quality_score
        except Exception as e:
            logger.warning(f"Error in prediction: {e}")
            bugs = self._identify_bugs(features, 0.5, ast_features)
            return bugs, 0.5
    
    def _extract_features(
        self,
        code_snippet: str,
        ast_features: Dict[str, Any]
    ) -> List[float]:
        """
        Extract numerical features from code for XGBoost.
        
        Args:
            code_snippet: Code to analyze
            ast_features: AST features
            
        Returns:
            List of feature values
        """
        features = []
        
        lines = code_snippet.split('\n')
        features.append(float(len(lines)))
        features.append(float(len(code_snippet)))
        features.append(float(code_snippet.count('for') + code_snippet.count('while')))
        features.append(float(code_snippet.count('if') + code_snippet.count('else')))
        features.append(float(code_snippet.count('try') + code_snippet.count('except')))
        
        features.append(float(len(ast_features.get('functions', []))))
        features.append(float(ast_features.get('cyclomatic_complexity', 0)))
        features.append(float(len(ast_features.get('imports', []))))
        features.append(float(ast_features.get('max_nesting_depth', 0)))
        
        features.append(float(code_snippet.count('TODO')))
        features.append(float(code_snippet.count('FIXME')))
        features.append(float(code_snippet.count('pass')))
        
        return features
    
    def _identify_bugs(
        self,
        features: List[float],
        probability: float,
        ast_features: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """
        Identify specific bug patterns.
        
        Args:
            features: Extracted features
            probability: Bug probability
            ast_features: AST features
            
        Returns:
            List of identified bugs
        """
        bugs = []
        
        if ast_features.get('cyclomatic_complexity', 0) > 10:
            bugs.append({
                'type': 'high_complexity',
                'severity': 'medium',
                'description': 'Function has high cyclomatic complexity (>10)',
                'line': ast_features.get('complexity_line', 0)
            })
        
        if ast_features.get('max_nesting_depth', 0) > 5:
            bugs.append({
                'type': 'deep_nesting',
                'severity': 'low',
                'description': 'Code has excessive nesting depth (>5 levels)',
                'line': ast_features.get('nesting_line', 0)
            })
        
        unused_imports = ast_features.get('unused_imports', [])
        if unused_imports:
            bugs.append({
                'type': 'unused_imports',
                'severity': 'low',
                'description': f'Found {len(unused_imports)} unused imports',
                'imports': unused_imports
            })
        
        if probability > 0.7:
            bugs.append({
                'type': 'potential_bug',
                'severity': 'high',
                'description': 'High probability of bugs detected (>70%)',
                'confidence': probability
            })
        
        return bugs
    
    def _get_feature_names(self) -> List[str]:
        """Get feature names for the model."""
        return [
            'line_count',
            'char_count',
            'loop_count',
            'condition_count',
            'exception_handling',
            'function_count',
            'cyclomatic_complexity',
            'import_count',
            'max_nesting_depth',
            'todo_count',
            'fixme_count',
            'pass_count'
        ]
    
    def train(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_val: np.ndarray = None,
        y_val: np.ndarray = None
    ):
        """
        Train the XGBoost model.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
        """
        if self.model is None:
            logger.error("Model not initialized")
            return
        
        eval_set = [(X_val, y_val)] if X_val is not None else None
        
        try:
            self.model.fit(
                X_train, y_train,
                eval_set=eval_set,
                verbose=False
            )
            logger.info("XGBoost model training completed")
        except Exception as e:
            logger.error(f"Error training model: {e}")
    
    def save(self, path: str):
        """Save the model to disk."""
        if self.model:
            try:
                Path(path).parent.mkdir(parents=True, exist_ok=True)
                self.model.save_model(path)
                logger.info(f"Model saved to {path}")
            except Exception as e:
                logger.error(f"Error saving model: {e}")
