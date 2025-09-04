import re
import logging
from typing import List, Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_brand_mentions(text: str, brands: List[str]) -> List[str]:
    """
    Extracts brand mentions from text using regex patterns
    
    Args:
        text: Text to analyze
        brands: List of brands to look for
    
    Returns:
        List of mentioned brands
    """
    if not text:
        return []
    
    text_lower = text.lower()
    mentioned_brands = []
    
    for brand in brands:
        # Simple regex pattern for brand mention
        pattern = r'\b' + re.escape(brand.lower()) + r'\b'
        if re.search(pattern, text_lower):
            mentioned_brands.append(brand)
    
    return mentioned_brands

def analyze_text_entities(text: str, primary_brand: str, competitor_brands: List[str]) -> Dict[str, Any]:
    """
    Analyzes text for brand mentions and returns entity analysis
    
    Args:
        text: Text to analyze
        primary_brand: Primary brand to track
        competitor_brands: List of competitor brands
    
    Returns:
        Dictionary with entity analysis results
    """
    all_brands = [primary_brand] + competitor_brands
    mentioned_brands = extract_brand_mentions(text, all_brands)
    
    return {
        "mentioned_brands": mentioned_brands,
        "primary_brand_mentioned": primary_brand in mentioned_brands,
        "competitors_mentioned": [brand for brand in mentioned_brands if brand != primary_brand],
        "any_brand_mentioned": len(mentioned_brands) > 0
    }