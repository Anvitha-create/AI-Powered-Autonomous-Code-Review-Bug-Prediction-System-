"""Graph Neural Network for code structure analysis."""

import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class GraphNeuralNetwork(nn.Module):
    """GNN for analyzing code dependency graphs."""
    
    def __init__(self, config):
        """Initialize GNN."""
        super().__init__()
        self.config = config
        self.device = torch.device(config.device)
        
        hidden_dim = config.gnn_hidden_dim
        num_layers = config.gnn_num_layers
        
        self.layers = nn.ModuleList([
            nn.Linear(768 if i == 0 else hidden_dim, hidden_dim)
            for i in range(num_layers)
        ])
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.classifier = nn.Linear(hidden_dim, 2)
        
        self.to(self.device)
    
    def analyze(
        self,
        graph: Dict[str, Any],
        embeddings: torch.Tensor
    ) -> Tuple[float, List[Dict[str, Any]]]:
        """
        Analyze code using GNN.
        
        Args:
            graph: Code dependency graph
            embeddings: CodeBERT embeddings
            
        Returns:
            Tuple of (score, vulnerabilities)
        """
        try:
            # Handle embeddings dimension
            if embeddings.dim() == 1:
                embeddings = embeddings.unsqueeze(0)
            
            # Ensure embeddings has shape [batch, seq_len, hidden_dim]
            if embeddings.shape[-1] != 768:
                # Pad or project to 768 dimensions
                if embeddings.shape[-1] < 768:
                    padding = torch.zeros(
                        embeddings.shape[0],
                        embeddings.shape[1] if embeddings.dim() > 1 else 1,
                        768 - embeddings.shape[-1],
                        device=self.device
                    )
                    embeddings = torch.cat([embeddings, padding], dim=-1)
                else:
                    embeddings = embeddings[..., :768]
            
            # Forward pass through GNN
            x = embeddings.to(self.device)
            
            # Take mean pooling if sequence dimension exists
            if x.dim() > 2:
                x = x.mean(dim=1)
            
            for layer in self.layers:
                x = layer(x)
                x = self.relu(x)
                x = self.dropout(x)
            
            # Classification
            output = self.classifier(x)
            
            # Handle output dimensions
            if output.dim() > 1:
                output = output.mean(dim=0)
            
            score = torch.softmax(output, dim=-1)[0].item() if output.dim() > 0 else 0.5
            
            # Detect vulnerabilities
            vulnerabilities = self._detect_vulnerabilities(graph, score)
            
            return score, vulnerabilities
            
        except Exception as e:
            logger.warning(f"Error in GNN analysis: {e}")
            return 0.5, []
    
    def _build_adjacency_matrix(self, graph: Dict[str, Any]) -> List[List[int]]:
        """Build adjacency matrix from graph."""
        nodes = graph.get('nodes', [])
        edges = graph.get('edges', [])
        
        n = len(nodes) if nodes else 1
        adj = [[0] * n for _ in range(n)]
        
        for edge in edges:
            if 'source' in edge and 'target' in edge:
                try:
                    src_idx = next(i for i, n in enumerate(nodes) if n.get('name') == edge['source'])
                    tgt_idx = next(i for i, n in enumerate(nodes) if n.get('name') == edge['target'])
                    adj[src_idx][tgt_idx] = 1
                except StopIteration:
                    pass
        
        return adj
    
    def _detect_vulnerabilities(
        self,
        graph: Dict[str, Any],
        score: float
    ) -> List[Dict[str, Any]]:
        """Detect security vulnerabilities."""
        vulnerabilities = []
        
        complexity = graph.get('complexity', 0)
        if complexity > 15:
            vulnerabilities.append({
                'type': 'high_complexity',
                'severity': 'medium',
                'description': 'High complexity may indicate security issues'
            })
        
        if score < 0.3:
            vulnerabilities.append({
                'type': 'suspicious_pattern',
                'severity': 'high',
                'description': 'Suspicious code pattern detected by GNN'
            })
        
        return vulnerabilities
