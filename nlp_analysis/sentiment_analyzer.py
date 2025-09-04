import google.generativeai as genai
import os
import json
import time
import re
import logging
from typing import Dict, Any
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class GeminiSentimentAnalyzer:
    def __init__(self):
        api_key = os.getenv("GEMINI_API_KEY")
        if not api_key:
            logger.error("GEMINI_API_KEY not found in environment variables")
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        try:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-pro')
            logger.info("Gemini AI configured successfully")
        except Exception as e:
            logger.error(f"Failed to configure Gemini: {e}")
            raise
    
    def analyze_sentiment(self, text: str, context: str = "", retries: int = 3) -> Dict[str, Any]:
        """
        Uses Gemini AI to analyze sentiment and extract themes from text
        
        Args:
            text: Text to analyze
            context: Additional context for the analysis
            retries: Number of retry attempts
        
        Returns:
            Dictionary with sentiment analysis results
        """
        prompt = f"""
        ACT as a expert marketing data analyst. Your task is to analyze the following text from a search result about 'smart fans'.
        Context: {context}
        Text to Analyze: \"\"\"{text}\"\"\"

        Analyze the text for:
        1. **Brand Mentions:** Identify if any of these brands are mentioned: [Atomberg, Havells, Crompton, Bajaj, Orpat, Usha].
        2. **Sentiment:** For each brand mentioned, classify the sentiment as 'Positive', 'Negative', or 'Neutral'. 
        3. **Key Themes:** Extract any key themes or topics discussed (e.g., 'price', 'design', 'bluetooth', 'noise', 'energy efficiency').

        Return your analysis ONLY as a valid JSON object with the following structure:
        {{
          "brand_mentions": ["Brand1", "Brand2"],
          "sentiment": {{
            "Brand1": "Positive",
            "Brand2": "Neutral"
          }},
          "themes": ["theme1", "theme2"]
        }}
        If no brands are mentioned, return "brand_mentions" as an empty list.
        """
        
        for attempt in range(retries):
            try:
                response = self.model.generate_content(prompt)
                # Extract JSON from Gemini's response
                json_str = response.text.strip().replace('```json', '').replace('```', '').replace('JSON', '')
                analysis_result = json.loads(json_str)
                return analysis_result
            except json.JSONDecodeError:
                # Try to extract JSON from malformed response
                try:
                    json_match = re.search(r'\{.*\}', response.text, re.DOTALL)
                    if json_match:
                        analysis_result = json.loads(json_match.group())
                        return analysis_result
                except:
                    if attempt < retries - 1:
                        time.sleep(2)  # Wait before retrying
                        continue
            except Exception as e:
                if attempt < retries - 1:
                    time.sleep(2)  # Wait before retrying
                    continue
        
        logger.error("Gemini analysis failed after multiple attempts")
        return self.default_analysis()
    
    def default_analysis(self) -> Dict[str, Any]:
        """Return a default analysis structure in case of API failure"""
        return {
            "brand_mentions": [],
            "sentiment": {},
            "themes": []
        }