# Atomberg SoV Analyzer

Streamlit app and Python scripts to compute Share of Voice (SoV) for Atomberg across Google and YouTube, with sentiment and engagement insights.

## Setup

1. Create and activate a virtual environment.
2. Install dependencies:
```
pip install -r requirements.txt
```
3. Create a `.env` file with keys:
```
SERPAPI_API_KEY=your_serpapi_key
YOUTUBE_API_KEY=your_youtube_data_api_key
GEMINI_API_KEY=your_gemini_api_key
```

## Run the Streamlit App
```
streamlit run main.py
```

## Orchestration Script
- Configure `config.yaml` at project root
- Run:
```
python orchestration_dag.py
```


## Notes
- If API keys are missing, the app falls back to mock data.
- Metrics implemented in `metrics_calculation.py` include volume, engagement, sentiment, and composite SoV.

