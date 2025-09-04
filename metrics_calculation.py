from typing import List, Dict, Any, Tuple
import pandas as pd


def _safe_bool(series: pd.Series) -> pd.Series:
	"""Convert a pandas Series to boolean safely, mapping missing to False."""
	if series is None:
		return pd.Series([], dtype=bool)
	try:
		return series.fillna(False).astype(bool)
	except Exception:
		return series.apply(lambda v: bool(v))


def _safe_float(series: pd.Series) -> pd.Series:
	"""Convert a pandas Series to float safely, mapping missing to 0.0."""
	if series is None:
		return pd.Series([], dtype=float)
	try:
		return pd.to_numeric(series.fillna(0.0), errors='coerce').fillna(0.0)
	except Exception:
		return series.apply(lambda v: float(v) if v is not None else 0.0)


def calculate_share_of_voice(analysis_results: List[Dict[str, Any]], primary_brand: str) -> Dict[str, float]:
	"""
	Calculate core Share of Voice metrics for a single keyword run.

	Expected fields in analysis_results rows (robust to missing):
	- source: str in {"google","youtube",...}
	- any_brand_mentioned: bool
	- primary_brand_mentioned: bool
	- engagement_score: float (YouTube)
	- sentiment_score: float (-1..1 for primary brand)
	"""
	df = pd.DataFrame(analysis_results)
	if len(df) == 0:
		return {
			"sov_volume": 0.0,
			"sov_engagement": 0.0,
			"total_results_analyzed": 0,
			"primary_brand_mentions": 0,
			"positive_mentions": 0,
			"negative_mentions": 0,
			"sentiment_ratio": 0.0,
			"share_of_positive_voice_primary": 0.0,
		}

	any_brand = _safe_bool(df.get('any_brand_mentioned', pd.Series(False, index=df.index)))
	primary_brand_mask = _safe_bool(df.get('primary_brand_mentioned', pd.Series(False, index=df.index)))
	sentiment = _safe_float(df.get('sentiment_score', pd.Series(0.0, index=df.index)))
	engagement = _safe_float(df.get('engagement_score', pd.Series(0.0, index=df.index)))
	source = df.get('source', pd.Series('', index=df.index)).fillna('')

	# Volume SoV
	total_mentions = int((any_brand).sum())
	primary_mentions = int((primary_brand_mask).sum())
	sov_volume = (primary_mentions / total_mentions * 100.0) if total_mentions > 0 else 0.0

	# Engagement SoV (YouTube only)
	youtube_mask = source.str.lower().eq('youtube')
	youtube_total_engagement = float(engagement[youtube_mask].sum())
	youtube_primary_engagement = float(engagement[youtube_mask & primary_brand_mask].sum())
	sov_engagement = (youtube_primary_engagement / youtube_total_engagement * 100.0) if youtube_total_engagement > 0 else 0.0

	# Sentiment for primary brand
	primary_sentiments = sentiment[primary_brand_mask]
	positive_mentions = int((primary_sentiments > 0.2).sum())
	negative_mentions = int((primary_sentiments < -0.2).sum())
	neutral_mentions = int(primary_mentions - positive_mentions - negative_mentions)
	sentiment_ratio = round(positive_mentions / (negative_mentions if negative_mentions > 0 else 1), 2)
	positive_rate_primary = (positive_mentions / primary_mentions * 100.0) if primary_mentions > 0 else 0.0

	return {
		"sov_volume": round(sov_volume, 2),
		"sov_engagement": round(sov_engagement, 2),
		"total_results_analyzed": int(len(df)),
		"primary_brand_mentions": primary_mentions,
		"positive_mentions": positive_mentions,
		"neutral_mentions": neutral_mentions,
		"negative_mentions": negative_mentions,
		"sentiment_ratio": sentiment_ratio,
		"share_of_positive_voice_primary": round(positive_rate_primary, 2),
	}


def platform_breakdown(analysis_results: List[Dict[str, Any]], primary_brand: str) -> Dict[str, Dict[str, float]]:
	"""Return per-platform SoV volume and engagement breakdowns."""
	df = pd.DataFrame(analysis_results)
	if len(df) == 0:
		return {}

	breakdown: Dict[str, Dict[str, float]] = {}
	for platform in df.get('source', pd.Series('')).fillna('').str.lower().unique():
		if platform == '':
			continue
		plat_df = df[df['source'].str.lower() == platform]
		any_brand = _safe_bool(plat_df.get('any_brand_mentioned', pd.Series(False, index=plat_df.index)))
		primary_brand_mask = _safe_bool(plat_df.get('primary_brand_mentioned', pd.Series(False, index=plat_df.index)))
		engagement = _safe_float(plat_df.get('engagement_score', pd.Series(0.0, index=plat_df.index)))

		total_mentions = int(any_brand.sum())
		primary_mentions = int(primary_brand_mask.sum())
		volume = (primary_mentions / total_mentions * 100.0) if total_mentions > 0 else 0.0

		if platform == 'youtube':
			total_eng = float(engagement.sum())
			prim_eng = float(engagement[primary_brand_mask].sum())
			eng = (prim_eng / total_eng * 100.0) if total_eng > 0 else 0.0
		else:
			eng = 0.0

		breakdown[platform] = {
			"sov_volume": round(volume, 2),
			"sov_engagement": round(eng, 2),
			"primary_mentions": primary_mentions,
			"total_mentions": total_mentions,
		}

	return breakdown


def weighted_sov(metrics: Dict[str, float], w_volume: float = 0.5, w_engagement: float = 0.3, w_positive: float = 0.2) -> float:
	"""
	Compute a single composite SoV score combining volume, engagement, and positive share.
	Weights must sum to 1.0.
	"""
	wv, we, wp = w_volume, w_engagement, w_positive
	if not abs((wv + we + wp) - 1.0) < 1e-6:
		# Normalize weights defensively
		total = (wv + we + wp) or 1.0
		wv, we, wp = wv / total, we / total, wp / total

	return round(
		wv * float(metrics.get('sov_volume', 0.0)) +
		we * float(metrics.get('sov_engagement', 0.0)) +
		wp * float(metrics.get('share_of_positive_voice_primary', 0.0)),
		2,
	)


def aggregate_multi_keyword(keyword_to_results: Dict[str, List[Dict[str, Any]]], primary_brand: str) -> Dict[str, Any]:
	"""
	Aggregate metrics across multiple keywords. Returns per-keyword metrics and overall averages.
	"""
	per_keyword: Dict[str, Any] = {}
	for kw, results in keyword_to_results.items():
		core = calculate_share_of_voice(results, primary_brand)
		per_keyword[kw] = {
			**core,
			"composite_sov": weighted_sov(core),
			"platform_breakdown": platform_breakdown(results, primary_brand),
		}

	# Overall summary
	if not per_keyword:
		return {"per_keyword": {}, "overall": {}}

	df = pd.DataFrame(per_keyword).T
	averages = {
		"sov_volume": round(float(df['sov_volume'].mean()), 2),
		"sov_engagement": round(float(df['sov_engagement'].mean()), 2),
		"share_of_positive_voice_primary": round(float(df['share_of_positive_voice_primary'].mean()), 2),
		"composite_sov": round(float(df['composite_sov'].mean()), 2),
	}

	return {"per_keyword": per_keyword, "overall": averages}
