"""Configuration management for the code review system."""

import os
from pathlib import Path
from dotenv import load_dotenv


class Config:
    """Configuration class for loading environment variables."""
    
    def __init__(self, env_file: str = ".env"):
        """Load configuration from .env file."""
        load_dotenv(env_file)
        
        # GitHub Configuration
        self.github_token = os.getenv('GITHUB_TOKEN', '')
        self.github_api_base = os.getenv('GITHUB_API_BASE', 'https://api.github.com')
        
        # Model Configuration
        self.model_path = os.getenv('MODEL_PATH', './models')
        self.codebert_model = os.getenv('CODEBERT_MODEL', 'microsoft/codebert-base')
        self.device = os.getenv('DEVICE', 'cpu')
        
        # XGBoost Configuration
        self.xgboost_model_path = os.getenv('XGBOOST_MODEL_PATH', './models/xgboost_model.pkl')
        self.xgboost_threshold = float(os.getenv('XGBOOST_THRESHOLD', '0.5'))
        
        # GNN Configuration
        self.gnn_hidden_dim = int(os.getenv('GNN_HIDDEN_DIM', '128'))
        self.gnn_num_layers = int(os.getenv('GNN_NUM_LAYERS', '3'))
        
        # LSTM Configuration
        self.lstm_hidden_dim = int(os.getenv('LSTM_HIDDEN_DIM', '256'))
        self.lstm_num_layers = int(os.getenv('LSTM_NUM_LAYERS', '2'))
        
        # Multi-Agent RL Configuration
        self.num_agents = int(os.getenv('NUM_AGENTS', '5'))
        self.learning_rate = float(os.getenv('LEARNING_RATE', '0.001'))
        self.gamma = float(os.getenv('GAMMA', '0.99'))
        
        # Logging
        self.log_level = os.getenv('LOG_LEVEL', 'INFO')
        self.log_file = os.getenv('LOG_FILE', './logs/code_review.log')
        
        # API Configuration
        self.api_host = os.getenv('API_HOST', '0.0.0.0')
        self.api_port = int(os.getenv('API_PORT', '8000'))
        self.debug = os.getenv('DEBUG', 'False').lower() == 'true'
        
        # Create necessary directories
        Path(self.model_path).mkdir(parents=True, exist_ok=True)
        Path(self.log_file).parent.mkdir(parents=True, exist_ok=True)