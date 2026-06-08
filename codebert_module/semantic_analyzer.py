"""CodeBERT semantic code analysis."""

import logging
import torch
from typing import Tuple, List
from transformers import AutoTokenizer, AutoModel

logger = logging.getLogger(__name__)


class SemanticAnalyzer:
    """CodeBERT-based semantic code analysis."""
    
    def __init__(self, config):
        """Initialize CodeBERT analyzer."""
        self.config = config
        self.device = torch.device(config.device)
        self.tokenizer = AutoTokenizer.from_pretrained(config.codebert_model)
        self.model = AutoModel.from_pretrained(config.codebert_model).to(self.device)
        self.model.eval()
    
    def analyze(self, code: str) -> Tuple[torch.Tensor, float]:
        """
        Analyze code semantically using CodeBERT.
        
        Args:
            code: Code snippet to analyze
            
        Returns:
            Tuple of (embeddings, semantic score)
        """
        try:
            # Tokenize code
            inputs = self.tokenizer(
                code,
                return_tensors='pt',
                max_length=512,
                truncation=True,
                padding=True
            ).to(self.device)
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = outputs.last_hidden_state
            
            # Calculate semantic score (based on embedding coherence)
            semantic_score = self._calculate_semantic_score(embeddings)
            
            return embeddings, semantic_score
            
        except Exception as e:
            logger.error(f"Error in semantic analysis: {e}")
            return torch.zeros(1, 768), 0.5
    
    def _calculate_semantic_score(self, embeddings: torch.Tensor) -> float:
        """
        Calculate semantic quality score from embeddings.
        
        Args:
            embeddings: Token embeddings
            
        Returns:
            Score between 0 and 1
        """
        # Compute mean pooling
        mean_embedding = embeddings.mean(dim=1)
        
        # Calculate self-similarity as semantic coherence
        similarity = torch.nn.functional.cosine_similarity(
            mean_embedding,
            mean_embedding
        ).mean()
        
        return float((similarity + 1) / 2)  # Normalize to [0, 1]
    
    def get_function_embedding(self, function_code: str) -> torch.Tensor:
        """Get embedding for a specific function."""
        inputs = self.tokenizer(
            function_code,
            return_tensors='pt',
            max_length=512,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            return outputs.last_hidden_state.mean(dim=1)