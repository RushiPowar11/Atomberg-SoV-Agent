import os
import requests
import logging
from typing import List, Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def fetch_youtube_videos(keyword: str, max_results: int = 20) -> List[Dict[str, Any]]:
    """
    Fetches YouTube videos using YouTube Data API v3
    
    Args:
        keyword: Search keyword
        max_results: Maximum number of results to fetch
    
    Returns:
        List of video details and statistics
    """
    api_key = os.getenv("YOUTUBE_API_KEY")
    if not api_key:
        logger.error("YOUTUBE_API_KEY not found in environment variables")
        raise ValueError("YOUTUBE_API_KEY not found in environment variables")
    
    try:
        # Search for videos
        search_url = "https://www.googleapis.com/youtube/v3/search"
        search_params = {
            "part": "snippet",
            "q": keyword,
            "type": "video",
            "maxResults": max_results,
            "key": api_key
        }
        
        logger.info(f"Fetching YouTube results for keyword: {keyword}")
        search_response = requests.get(search_url, params=search_params, timeout=30)
        search_response.raise_for_status()
        search_data = search_response.json()
        
        video_ids = [item["id"]["videoId"] for item in search_data.get("items", [])]
        
        if not video_ids:
            logger.warning("No YouTube videos found for the keyword")
            return []
        
        # Get video statistics
        stats_url = "https://www.googleapis.com/youtube/v3/videos"
        stats_params = {
            "part": "snippet,statistics,contentDetails",
            "id": ",".join(video_ids),
            "key": api_key
        }
        
        stats_response = requests.get(stats_url, params=stats_params, timeout=30)
        stats_response.raise_for_status()
        stats_data = stats_response.json()
        
        results = []
        for item in stats_data.get("items", []):
            results.append({
                "title": item["snippet"]["title"],
                "link": f"https://www.youtube.com/watch?v={item['id']}",
                "description": item["snippet"]["description"],
                "channel": item["snippet"]["channelTitle"],
                "views": int(item["statistics"].get("viewCount", 0)),
                "likes": int(item["statistics"].get("likeCount", 0)),
                "comments": int(item["statistics"].get("commentCount", 0)),
                "duration": item["contentDetails"].get("duration", "PT0M0S"),
                "published_at": item["snippet"].get("publishedAt", ""),
                "source": "youtube"
            })
        
        logger.info(f"Fetched {len(results)} YouTube results")
        return results
        
    except requests.exceptions.RequestException as e:
        logger.error(f"Error fetching YouTube results: {e}")
        raise
    except Exception as e:
        logger.error(f"Unexpected error in YouTube search: {e}")
        raise