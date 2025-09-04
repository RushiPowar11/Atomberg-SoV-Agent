#!/usr/bin/env python3
"""
Orchestration script for Atomberg SoV analysis
Can be scheduled to run periodically (e.g., with cron or Airflow)
"""

import yaml
import json
import logging
from datetime import datetime
from data_collection.google_search import fetch_google_search_results
from data_collection.youtube_search import fetch_youtube_videos
from nlp_analysis.entity_recognizer import analyze_text_entities
from nlp_analysis.sentiment_analyzer import GeminiSentimentAnalyzer
from utils.helpers import calculate_engagement_score, parse_duration

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sov_analysis.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def load_config():
    """Load configuration from YAML file at project root."""
    try:
        with open('config.yaml', 'r') as f:
            return yaml.safe_load(f)
    except FileNotFoundError:
        logger.error("Configuration file not found at project root: config.yaml")
        raise

def main():
    """Main orchestration function"""
    logger.info("Starting Atomberg SoV Analysis")
    
    # Load configuration
    config = load_config()
    
    # Initialize sentiment analyzer
    try:
        sentiment_analyzer = GeminiSentimentAnalyzer()
    except Exception as e:
        logger.error(f"Failed to initialize sentiment analyzer: {e}")
        return
    
    # Fetch data from all platforms
    all_results = []
    
    if "google" in config['search']['platforms']:
        try:
            logger.info(f"Fetching Google results for: {config['search']['keyword']}")
            google_results = fetch_google_search_results(
                config['search']['keyword'],
                config['search']['num_results']
            )
            all_results.extend(google_results)
        except Exception as e:
            logger.error(f"Failed to fetch Google results: {e}")
    
    if "youtube" in config['search']['platforms']:
        try:
            logger.info(f"Fetching YouTube results for: {config['search']['keyword']}")
            youtube_results = fetch_youtube_videos(
                config['search']['keyword'],
                config['search']['num_results']
            )
            # Add engagement score for YouTube videos
            for video in youtube_results:
                video['engagement_score'] = calculate_engagement_score(
                    video['views'], video['likes'], video['comments']
                )
                video['duration_min'] = parse_duration(video['duration'])
            all_results.extend(youtube_results)
        except Exception as e:
            logger.error(f"Failed to fetch YouTube results: {e}")
    
    if not all_results:
        logger.warning("No results found. Exiting.")
        return
    
    # Analyze results
    analysis_results = []
    for result in all_results:
        # Get text to analyze based on platform
        text_to_analyze = result.get('snippet') or result.get('description') or result.get('title', '')
        
        # Entity recognition
        entity_analysis = analyze_text_entities(
            text_to_analyze,
            config['brands']['primary'],
            config['brands']['competitors']
        )
        
        # Sentiment analysis
        sentiment_analysis = sentiment_analyzer.analyze_sentiment(
            text_to_analyze,
            f"Search result for {config['search']['keyword']}"
        )
        
        # Process sentiment for primary brand
        primary_brand_sentiment = sentiment_analysis.get("sentiment", {}).get(config['brands']['primary'], "Neutral")
        sentiment_score = 0
        if primary_brand_sentiment == "Positive":
            sentiment_score = 1
        elif primary_brand_sentiment == "Negative":
            sentiment_score = -1
        
        analysis_results.append({
            **result,
            **entity_analysis,
            'primary_brand_sentiment': primary_brand_sentiment,
            'sentiment_score': sentiment_score,
            'themes': sentiment_analysis.get('themes', []),
            'analysis_timestamp': datetime.now().isoformat()
        })
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    try:
        with open(f'outputs/analysis_{timestamp}.json', 'w') as f:
            json.dump(analysis_results, f, indent=2)
        logger.info(f"Analysis results saved to outputs/analysis_{timestamp}.json")
    except Exception as e:
        logger.error(f"Failed to save results: {e}")
    
    logger.info("Atomberg SoV Analysis completed successfully")

if __name__ == "__main__":
    main()