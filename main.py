import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import json
import requests
from datetime import datetime
import google.generativeai as genai
import os
from dotenv import load_dotenv
import re
from typing import List, Dict, Any
from metrics_calculation import calculate_share_of_voice, aggregate_multi_keyword

# Prefer real fetchers; fallback to local mock functions if unavailable or misconfigured
try:
	from data_collection.google_search import fetch_google_search_results as real_fetch_google
except Exception:
	real_fetch_google = None

try:
	from data_collection.youtube_search import fetch_youtube_videos as real_fetch_youtube
except Exception:
	real_fetch_youtube = None

try:
	from nlp_analysis.entity_recognizer import analyze_text_entities as entity_analyze
except Exception:
	entity_analyze = None

try:
	from nlp_analysis.sentiment_analyzer import GeminiSentimentAnalyzer
except Exception:
	GeminiSentimentAnalyzer = None

try:
	from utils.helpers import calculate_engagement_score
except Exception:
	# Fallback defined later if import fails
	calculate_engagement_score = None

# Load environment variables
load_dotenv()

# Configure the app
st.set_page_config(
    page_title="Atomberg SoV Analyzer",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize Gemini API
def configure_gemini():
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        st.error("GEMINI_API_KEY not found in environment variables. Please add it to your .env file.")
        return None
    
    try:
        genai.configure(api_key=api_key)
        return genai.GenerativeModel('gemini-2.0-flash')
    except Exception as e:
        st.error(f"Failed to configure Gemini: {e}")
        return None

# Mock functions for data acquisition (fallback if real fetchers not available)
def fetch_google_search_results_mock(keyword: str, num_results: int = 10) -> List[Dict[str, Any]]:
    """Mock function to simulate Google search results"""
    # In a real implementation, this would use SerpAPI or Google Custom Search API
    st.info("Using mock data for demonstration. Replace with actual API calls.")
    
    # Sample mock data
    mock_results = [
        {
            'title': 'Atomberg Studios Smart Fan Review - Best in Class',
            'link': 'https://example.com/atomberg-review',
            'snippet': 'Atomberg Studios smart fans offer incredible energy efficiency with their BLDC technology. Compared to Havells and Crompton, Atomberg saves up to 65% on electricity bills.',
            'position': 1,
            'source': 'google'
        },
        {
            'title': 'Top 5 Smart Fans in India 2023 - Comparison',
            'link': 'https://example.com/top-smart-fans',
            'snippet': 'Havells, Crompton, and Atomberg are the top contenders in the smart fan market. Bajaj and Orpat also offer competitive models with basic smart features.',
            'position': 2,
            'source': 'google'
        },
        {
            'title': 'Smart Fan Buying Guide - What to Consider',
            'link': 'https://example.com/buying-guide',
            'snippet': 'When choosing between Atomberg, Havells, and other brands, consider energy efficiency, smart features, and design. Atomberg leads in energy savings.',
            'position': 3,
            'source': 'google'
        },
        {
            'title': 'Havells Smart Fan vs Atomberg - Detailed Comparison',
            'link': 'https://example.com/havells-vs-atomberg',
            'snippet': 'Havells offers better availability in retail stores, but Atomberg wins in terms of technological innovation and energy efficiency.',
            'position': 4,
            'source': 'google'
        },
        {
            'title': 'Crompton Smart Fan Features and Pricing',
            'link': 'https://example.com/crompton-smart-fan',
            'snippet': 'Crompton has entered the smart fan market with models that compete with Atomberg and Havells. Their pricing is competitive but features are limited.',
            'position': 5,
            'source': 'google'
        }
    ]
    
    # Filter by keyword in title or snippet
    filtered_results = [r for r in mock_results if keyword.lower() in r['snippet'].lower() or keyword.lower() in r['title'].lower()]
    return filtered_results[:num_results]

def fetch_youtube_videos_mock(keyword: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """Mock function to simulate YouTube search results"""
    # In a real implementation, this would use YouTube Data API
    st.info("Using mock data for demonstration. Replace with actual API calls.")
    
    # Sample mock data
    mock_videos = [
        {
            'title': 'Atomberg Smart Fan Unboxing and Review',
            'link': 'https://youtube.com/watch?v=atomberg123',
            'description': 'Detailed review of Atomberg smart fan features, installation process, and performance comparison with traditional fans.',
            'views': 150000,
            'likes': 4500,
            'comments': 320,
            'source': 'youtube'
        },
        {
            'title': 'Havells vs Atomberg - Which Smart Fan is Better?',
            'link': 'https://youtube.com/watch?v=havells-vs-atomberg',
            'description': 'Comparison between Havells and Atomberg smart fans. We look at design, features, power consumption, and value for money.',
            'views': 220000,
            'likes': 6200,
            'comments': 480,
            'source': 'youtube'
        },
        {
            'title': 'How Atomberg Smart Fans Save Electricity',
            'link': 'https://youtube.com/watch?v=atomberg-savings',
            'description': 'Technical explanation of how Atomberg BLDC technology reduces power consumption compared to Bajaj, Crompton and other brands.',
            'views': 98000,
            'likes': 3100,
            'comments': 210,
            'source': 'youtube'
        },
        {
            'title': 'Top 5 Smart Fans for Your Home - 2023 Edition',
            'link': 'https://youtube.com/watch?v=top5-smart-fans',
            'description': 'We review the best smart fans from Atomberg, Havells, Crompton, Bajaj, and Orpat. See which one fits your needs and budget.',
            'views': 185000,
            'likes': 5200,
            'comments': 390,
            'source': 'youtube'
        }
    ]
    
    # Filter by keyword in title or description
    filtered_videos = [v for v in mock_videos if keyword.lower() in v['description'].lower() or keyword.lower() in v['title'].lower()]
    return filtered_videos[:max_results]

# Analysis function using Gemini AI
def analyze_with_gemini(context, text_to_analyze):
    """Analyze text using Gemini AI for sentiment and brand mentions"""
    model = configure_gemini()
    if not model:
        return default_analysis()
    
    prompt = f"""
    ACT as a expert marketing data analyst. Your task is to analyze the following text from a search result about 'smart fans'.
    Context: {context}
    Text to Analyze: \"\"\"{text_to_analyze}\"\"\"

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
    
    try:
        response = model.generate_content(prompt)
        # Extract JSON from Gemini's response
        json_str = response.text.strip().replace('```json', '').replace('```', '')
        analysis_result = json.loads(json_str)
        return analysis_result
    except Exception as e:
        st.error(f"Gemini analysis failed: {e}")
        return default_analysis()

def analyze_with_gemini_central(text: str, context: str = "") -> Dict[str, Any]:
    """Prefer centralized GeminiSentimentAnalyzer if available."""
    if GeminiSentimentAnalyzer is None:
        return analyze_with_gemini(context, text)
    try:
        analyzer = GeminiSentimentAnalyzer()
        return analyzer.analyze_sentiment(text, context)
    except Exception:
        return analyze_with_gemini(context, text)

def default_analysis():
    """Return a default analysis structure in case of API failure"""
    return {
        "brand_mentions": [],
        "sentiment": {},
        "themes": []
    }

# Calculate engagement score for YouTube videos
if calculate_engagement_score is None:
    def calculate_engagement_score(views, likes, comments):
        """Simple fallback engagement score calculation."""
        try:
            views = int(views or 0)
            likes = int(likes or 0)
            comments = int(comments or 0)
            if views > 0:
                engagement_rate = (likes + comments * 5) / views
                return engagement_rate * 1000
            return 0
        except Exception:
            return 0

# Main app function
def main():
    st.title("Atomberg Share of Voice Analyzer")
    st.markdown("Analyze Atomberg's presence and sentiment across Google and YouTube for any keyword")
    
    # Sidebar configuration
    st.sidebar.header("Configuration")
    keyword = st.sidebar.text_input("Enter Keyword", "smart fan")
    multi_keywords_raw = st.sidebar.text_area("Additional Keywords (comma-separated)", "smart ceiling fan, bldc fan, energy efficient fan")
    multi_keywords = [k.strip() for k in multi_keywords_raw.split(',') if k.strip()]
    platforms = st.sidebar.multiselect("Choose Platforms", ["Google", "YouTube"], default=["Google", "YouTube"])
    num_results = st.sidebar.slider("Number of Results", 5, 20, 10)
    
    # Brand selection
    st.sidebar.subheader("Brand Focus")
    primary_brand = st.sidebar.selectbox("Primary Brand", ["Atomberg", "Havells", "Crompton", "Bajaj"], index=0)
    competitors = st.sidebar.multiselect(
        "Competitors to Track", 
        ["Havells", "Crompton", "Bajaj", "Orpat", "Usha"], 
        default=["Havells", "Crompton", "Bajaj"]
    )
    
    if st.sidebar.button("Analyze", type="primary"):
        # Initialize session state for results
        if 'analysis_results' not in st.session_state:
            st.session_state.analysis_results = []
        if 'sov_data' not in st.session_state:
            st.session_state.sov_data = []
        
        # 1. Fetch Data
        with st.spinner("Fetching data from APIs..."):
            all_results = []
            if "Google" in platforms:
                try:
                    if real_fetch_google is not None:
                        google_results = real_fetch_google(keyword, num_results)
                    else:
                        google_results = fetch_google_search_results_mock(keyword, num_results)
                except Exception:
                    google_results = fetch_google_search_results_mock(keyword, num_results)
                all_results.extend(google_results)
            
            if "YouTube" in platforms:
                try:
                    if real_fetch_youtube is not None:
                        youtube_results = real_fetch_youtube(keyword, num_results)
                    else:
                        youtube_results = fetch_youtube_videos_mock(keyword, num_results)
                except Exception:
                    youtube_results = fetch_youtube_videos_mock(keyword, num_results)
                # Add engagement score for YouTube videos
                for video in youtube_results:
                    video['engagement_score'] = calculate_engagement_score(
                        video['views'], video['likes'], video['comments']
                    )
                all_results.extend(youtube_results)
        
        if not all_results:
            st.error("No results found. Try a different keyword or platform.")
            return
            
        sov_data = []
        
        # 2. Analyze each result with Gemini
        with st.spinner("Analyzing sentiment and mentions with AI..."):
            progress_bar = st.progress(0)
            for i, result in enumerate(all_results):
                # Get text to analyze based on platform
                text_to_analyze = result.get('snippet') or result.get('description') or result.get('title', '')
                
                # Prefer centralized Gemini analyzer when available
                analysis = analyze_with_gemini_central(text_to_analyze, f"Search Result for '{keyword}'")
                
                # Process analysis
                # Use regex-based entity extraction if available to complement LLM
                if entity_analyze is not None:
                    entity = entity_analyze(text_to_analyze, primary_brand, competitors)
                    atomberg_mentioned = entity.get('primary_brand_mentioned', False)
                    competitors_mentioned = len(entity.get('competitors_mentioned', [])) > 0
                    any_brand_mentioned = entity.get('any_brand_mentioned', False)
                else:
                    atomberg_mentioned = "Atomberg" in analysis.get("brand_mentions", [])
                    competitors_mentioned = any(brand in analysis.get("brand_mentions", []) for brand in competitors)
                    any_brand_mentioned = atomberg_mentioned or competitors_mentioned
                
                # Get sentiment for Atomberg if mentioned
                atomberg_sentiment = analysis.get("sentiment", {}).get("Atomberg", "Neutral")
                
                # Calculate sentiment score
                sentiment_score = 0
                if atomberg_sentiment == "Positive":
                    sentiment_score = 1
                elif atomberg_sentiment == "Negative":
                    sentiment_score = -1
                
                sov_data.append({
                    'title': result.get('title', ''),
                    'source': result.get('source', ''),
                    'primary_brand_mentioned': atomberg_mentioned,
                    'competitors_mentioned': competitors_mentioned,
                    'any_brand_mentioned': any_brand_mentioned,
                    'atomberg_sentiment': atomberg_sentiment,
                    'sentiment_score': sentiment_score,
                    'engagement_score': result.get('engagement_score', 0),
                    'themes': analysis.get('themes', [])
                })
                
                progress_bar.progress((i + 1) / len(all_results))
            
            st.session_state.sov_data = sov_data
            st.session_state.analysis_results = all_results
        
        st.success("Analysis complete!")
    
    # Display results if available
    if 'sov_data' in st.session_state and st.session_state.sov_data:
        sov_data = st.session_state.sov_data
        analysis_results = st.session_state.analysis_results
        
        # Calculate metrics
        df = pd.DataFrame(sov_data)
        
        # Core SoV metrics using metrics module
        metrics = calculate_share_of_voice(df.to_dict(orient='records'), primary_brand)
        sov_percentage = metrics.get('sov_volume', 0)
        atomberg_mentions = metrics.get('primary_brand_mentions', 0)
        
        sov_engagement = metrics.get('sov_engagement', 0)
        
        # Sentiment metrics
        positive_mentions = metrics.get('positive_mentions', 0)
        negative_mentions = metrics.get('negative_mentions', 0)
        neutral_mentions = metrics.get('neutral_mentions', 0)
        
        # Display metrics
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Atomberg Share of Voice", f"{sov_percentage:.1f}%")
        with col2:
            st.metric("Engagement Share", f"{sov_engagement:.1f}%")
        with col3:
            st.metric("Share of Positive Voice", f"{metrics.get('share_of_positive_voice_primary', 0):.1f}%")
        
        # Visualization
        st.subheader("Visual Analysis")
        
        # Create tabs for different visualizations
        tab1, tab2, tab3, tab4 = st.tabs(["Share of Voice", "Sentiment Analysis", "Theme Analysis", "Raw Data"])
        
        with tab1:
            # Share of Voice chart
            atomberg_color = '#1f77b4'  # blue
            competitors_color = '#ff7f0e'  # orange
            fig_sov = go.Figure(data=[
                go.Bar(
                    name='Atomberg',
                    x=['Volume', 'Engagement'],
                    y=[sov_percentage, sov_engagement],
                    marker_color=atomberg_color,
                    legendgroup='atomberg'
                ),
                go.Bar(
                    name='Competitors',
                    x=['Volume', 'Engagement'],
                    y=[max(0, 100-sov_percentage), max(0, 100-sov_engagement)],
                    marker_color=competitors_color,
                    legendgroup='competitors'
                )
            ])
            fig_sov.update_layout(
                barmode='stack',
                title='Share of Voice: Atomberg vs Competitors',
                yaxis_title='Percentage (%)'
            )
            st.plotly_chart(fig_sov, use_container_width=True)
        
        with tab2:
            # Sentiment analysis chart
            if atomberg_mentions > 0:
                sentiment_counts = [positive_mentions, neutral_mentions, negative_mentions]
                sentiment_labels = ['Positive', 'Neutral', 'Negative']
                fig_sentiment = px.pie(
                    values=sentiment_counts, 
                    names=sentiment_labels, 
                    title='Atomberg Sentiment Distribution',
                    color=sentiment_labels,
                    color_discrete_map={'Positive': '#2ca02c', 'Neutral': '#7f7f7f', 'Negative': '#d62728'}
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)
            else:
                st.info("No Atomberg mentions to analyze sentiment")
        
        with tab3:
            # Theme analysis
            all_themes = []
            for themes in df['themes']:
                all_themes.extend(themes)
            
            if all_themes:
                theme_series = pd.Series(all_themes)
                theme_counts = theme_series.value_counts().head(10)
                
                fig_themes = px.bar(
                    x=theme_counts.values, 
                    y=theme_counts.index, 
                    orientation='h',
                    title='Top Themes in Search Results'
                )
                fig_themes.update_layout(yaxis={'categoryorder':'total ascending'})
                st.plotly_chart(fig_themes, use_container_width=True)
            else:
                st.info("No themes extracted from the analysis")
        
        with tab4:
            # Raw data display
            raw_cols = [c for c in ['title', 'source', 'primary_brand_mentioned', 'atomberg_sentiment', 'engagement_score'] if c in df.columns]
            st.dataframe(df[raw_cols])

        # Multi-keyword aggregation (if provided)
        if multi_keywords:
            st.subheader("Cross-Keyword Insights")
            kw_to_results: Dict[str, List[Dict[str, Any]] ] = {"primary": sov_data}
            # For each additional keyword, fetch and analyze quickly (shallow to save time)
            with st.spinner("Fetching and analyzing additional keywords..."):
                for kw in multi_keywords:
                    tmp_results = []
                    try:
                        g = real_fetch_google(kw, max(5, num_results // 2)) if real_fetch_google else fetch_google_search_results_mock(kw, max(5, num_results // 2))
                    except Exception:
                        g = fetch_google_search_results_mock(kw, max(5, num_results // 2))
                    try:
                        y = real_fetch_youtube(kw, max(5, num_results // 2)) if real_fetch_youtube else fetch_youtube_videos_mock(kw, max(5, num_results // 2))
                    except Exception:
                        y = fetch_youtube_videos_mock(kw, max(5, num_results // 2))
                    for video in y:
                        video['engagement_score'] = calculate_engagement_score(video.get('views', 0), video.get('likes', 0), video.get('comments', 0))
                    for r in (g + y):
                        text = r.get('snippet') or r.get('description') or r.get('title', '')
                        analysis = analyze_with_gemini_central(text, f"Search Result for '{kw}'")
                        if entity_analyze is not None:
                            entity = entity_analyze(text, primary_brand, competitors)
                            prim = entity.get('primary_brand_mentioned', False)
                            comp = len(entity.get('competitors_mentioned', [])) > 0
                            anym = entity.get('any_brand_mentioned', False)
                        else:
                            prim = primary_brand in analysis.get('brand_mentions', [])
                            comp = any(b in analysis.get('brand_mentions', []) for b in competitors)
                            anym = prim or comp
                        senti = analysis.get('sentiment', {}).get(primary_brand, 'Neutral')
                        score = 1 if senti == 'Positive' else -1 if senti == 'Negative' else 0
                        tmp_results.append({
                            'title': r.get('title',''),
                            'source': r.get('source',''),
                            'primary_brand_mentioned': prim,
                            'competitors_mentioned': comp,
                            'any_brand_mentioned': anym,
                            'atomberg_sentiment': senti,
                            'sentiment_score': score,
                            'engagement_score': r.get('engagement_score', 0),
                        })
                    kw_to_results[kw] = tmp_results

            agg = aggregate_multi_keyword(kw_to_results, primary_brand)
            overall = agg.get('overall', {})
            st.metric("Cross-Keyword Composite SoV", f"{overall.get('composite_sov', 0):.1f}")
            st.write("Per-keyword summary:")
            per_kw_df = pd.DataFrame(agg.get('per_keyword', {})).T
            st.dataframe(per_kw_df)
        
        # Recommendations section
        st.subheader("Recommendations for Atomberg")
        
        # Generate recommendations based on analysis
        recommendations = []
        
        if sov_percentage < 30:
            recommendations.append(
                "Increase content marketing efforts to improve Share of Voice. Consider creating more comparison content that highlights Atomberg's advantages."
            )
        
        if sov_engagement > sov_percentage:
            recommendations.append(
                "Leverage the higher engagement rate by partnering with influencers who can create authentic content about Atomberg products."
            )
        else:
            recommendations.append(
                "Focus on creating more engaging content. Video demonstrations and customer testimonials could help improve engagement rates."
            )
        
        if negative_mentions > 0:
            recommendations.append(
                "Address negative sentiment by highlighting product improvements and showcasing positive customer experiences."
            )
        
        if not recommendations:
            recommendations.append(
                "Current performance is strong. Maintain content quality and consider expanding to additional platforms."
            )
        
        for i, rec in enumerate(recommendations, 1):
            st.markdown(f"{i}. {rec}")
    
    else:
        # Show instructions if no analysis has been run
        st.info(
            """
            **Instructions:**
            1. Configure your search using the sidebar options
            2. Select platforms to analyze (Google and/or YouTube)
            3. Click the 'Analyze' button to run the analysis
            4. View results and recommendations in the main panel
            
            *Note: This demo uses mock data. In a production environment, you would integrate with actual APIs.*
            """
        )
        
        # Placeholder charts
        st.subheader("Example Analysis Output")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Placeholder SoV chart
            fig = go.Figure(data=[
                go.Bar(name='Atomberg', x=['Volume', 'Engagement'], y=[35, 42]),
                go.Bar(name='Competitors', x=['Volume', 'Engagement'], y=[65, 58])
            ])
            fig.update_layout(
                barmode='stack',
                title='Share of Voice: Atomberg vs Competitors',
                yaxis_title='Percentage (%)'
            )
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            # Placeholder sentiment chart
            fig2 = px.pie(
                values=[65, 25, 10], 
                names=['Positive', 'Neutral', 'Negative'], 
                title='Atomberg Sentiment Distribution'
            )
            st.plotly_chart(fig2, use_container_width=True)

if __name__ == "__main__":
    main()