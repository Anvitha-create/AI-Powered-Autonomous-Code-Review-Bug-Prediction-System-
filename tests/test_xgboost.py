"""Unit tests for XGBoost module."""

import pytest
from xgboost_module import BugPredictor
from utils import Config


class TestBugPredictor:
    """Test cases for bug prediction."""
    
    @pytest.fixture
    def config(self):
        return Config()
    
    @pytest.fixture
    def predictor(self, config):
        return BugPredictor(config)
    
    def test_feature_extraction(self, predictor):
        """Test feature extraction."""
        code = "def test():\n    x = 1\n    return x"
        ast_features = {
            'functions': [{'name': 'test'}],
            'cyclomatic_complexity': 1,
            'imports': [],
            'max_nesting_depth': 1
        }
        features = predictor._extract_features(code, ast_features)
        assert len(features) > 0
    
    def test_predict(self, predictor):
        """Test prediction."""
        code = "def test():\n    x = 1"
        ast_features = {
            'functions': [{'name': 'test'}],
            'cyclomatic_complexity': 1,
            'imports': [],
            'max_nesting_depth': 1
        }
        bugs, score = predictor.predict(code, ast_features)
        assert isinstance(bugs, list)
        assert 0 <= score <= 1