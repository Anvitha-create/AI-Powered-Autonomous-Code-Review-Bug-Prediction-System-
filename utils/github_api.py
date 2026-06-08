"""GitHub API integration."""

import logging
from typing import Dict, Any, List
import requests
from requests.auth import HTTPBasicAuth

logger = logging.getLogger(__name__)


class GitHubAPI:
    """GitHub API wrapper for repository access."""
    
    def __init__(self, token: str):
        """
        Initialize GitHub API client.
        
        Args:
            token: GitHub personal access token
        """
        self.token = token
        self.headers = {
            'Authorization': f'token {token}',
            'Accept': 'application/vnd.github.v3+json'
        }
        self.base_url = 'https://api.github.com'
    
    def get_pr_files(self, owner: str, repo: str, pr_number: int) -> List[Dict[str, Any]]:
        """
        Get files changed in a pull request.
        
        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number
            
        Returns:
            List of file information
        """
        url = f'{self.base_url}/repos/{owner}/{repo}/pulls/{pr_number}/files'
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Error fetching PR files: {e}")
            return []
    
    def get_file_content(self, owner: str, repo: str, path: str, ref: str = 'main') -> str:
        """
        Get file content from repository.
        
        Args:
            owner: Repository owner
            repo: Repository name
            path: File path
            ref: Branch/commit reference
            
        Returns:
            File content
        """
        url = f'{self.base_url}/repos/{owner}/{repo}/contents/{path}?ref={ref}'
        try:
            response = requests.get(url, headers=self.headers)
            response.raise_for_status()
            import base64
            return base64.b64decode(response.json()['content']).decode('utf-8')
        except requests.RequestException as e:
            logger.error(f"Error fetching file content: {e}")
            return ""
    
    def post_pr_comment(
        self,
        owner: str,
        repo: str,
        pr_number: int,
        comment: str
    ) -> bool:
        """
        Post a comment on a pull request.
        
        Args:
            owner: Repository owner
            repo: Repository name
            pr_number: Pull request number
            comment: Comment text
            
        Returns:
            True if successful
        """
        url = f'{self.base_url}/repos/{owner}/{repo}/issues/{pr_number}/comments'
        try:
            response = requests.post(
                url,
                headers=self.headers,
                json={'body': comment}
            )
            response.raise_for_status()
            logger.info("Comment posted successfully")
            return True
        except requests.RequestException as e:
            logger.error(f"Error posting comment: {e}")
            return False