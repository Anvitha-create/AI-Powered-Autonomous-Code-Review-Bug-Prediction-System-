"""LSTM for sequential pattern detection."""

import torch
import torch.nn as nn
from typing import List
import logging

logger = logging.getLogger(__name__)


class SequenceAnalyzer(nn.Module):
    """LSTM-based sequential anomaly detection."""
    
    def __init__(self, config):
        """Initialize LSTM analyzer."""
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        input_dim = 128
        hidden_dim = config.lstm_hidden_dim
        num_layers = config.lstm_num_layers
        
        self.embedding = nn.Embedding(256, input_dim)
        self.lstm = nn.LSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=0.2
        )
        
        self.fc = nn.Linear(hidden_dim, 2)
        self.to(self.device)
    
    def detect_anomalies(self, code: str) -> float:
        """
        Detect anomalies in code sequences.
        
        Args:
            code: Code snippet
            
        Returns:
            Anomaly score
        """
        try:
            # Convert code to token sequences
            tokens = self._code_to_tokens(code)
            
            if len(tokens) == 0:
                return 0.5
            
            # Pad/truncate to fixed length
            max_len = 512
            tokens = tokens[:max_len]
            tokens += [0] * (max_len - len(tokens))
            
            # Forward pass
            x = torch.tensor([tokens], dtype=torch.long).to(self.device)
            embedded = self.embedding(x)
            
            lstm_out, _ = self.lstm(embedded)
            last_output = lstm_out[:, -1, :]
            
            output = self.fc(last_output)
            anomaly_score = torch.softmax(output, dim=1)[0][1].item()
            
            return anomaly_score
            
        except Exception as e:
            logger.error(f"Error in anomaly detection: {e}")
            return 0.5
    
    def _code_to_tokens(self, code: str) -> List[int]:
        """Convert code to token indices."""
        # Simple tokenization - map characters to indices
        token_map = {chr(i): i for i in range(32, 127)}
        tokens = []
        
        for char in code:
            if char in token_map:
                tokens.append(token_map[char])
            elif ord(char) < 256:
                tokens.append(ord(char))
        
        return tokens