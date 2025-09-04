import os
import requests
import logging
from typing import List, Dict, Any
from urllib.parse import quote_plus
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_google_search_results(keyword: str, num_results: int = 20) -> List[Dict[str, Any]]:
    """
    Fetches Google search results using SerpAPI
    
    Args:
        keyword: Search keyword
        num_results: Number of results to fetch
    
    Returns:
        List of search results with title, link, and snippet
    """
    api_key = os.getenv("SERPAPI_API_KEY")
    if not api_key:
        logger.error("SERPAPI_API_KEY not found in environment variables")
        raise ValueError("SERPAPI_API_KEY not found in environment variables")
    
    try:
        # Encode the keyword for URL
        encoded_keyword = quote_plus(keyword)
        
        # Make API request to SerpAPI
        params = {
            "q": encoded_keyword,
            "num": num_results,
            "api_key": api_key,
            "engine": "google"
        }
        
        logger.info(f"Fetching Google results for keyword: {keyword}")
        response = requests.get("https://serpapi.com/search", params=params, timeout=30)
        response.raise_for_status()
        data = response.json()
        
        results = []
        if "organic_results" in data:
            for item in data["organic_results"]:
                results.append({
                    "title": item.get("title", ""),
                    "link": item.get("link", ""),
                    "snippet": item.get("snippet", ""),
                    "position": item.get("position", 0),
                    "source": "google"
                })
        
        logger.info(f"Fetched {len(results)} Google results")
        return results
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching Google results: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in Google search: {e}")
        raise