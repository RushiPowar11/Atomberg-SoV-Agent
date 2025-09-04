import re


def calculate_engagement_score(views: int, likes: int, comments: int) -> float:
	"""Calculate a normalized engagement score for YouTube videos.

	The score is scaled by 1000 for better visualization and weights comments
	more heavily to reflect deeper engagement.
	"""
	try:
		views = int(views or 0)
		likes = int(likes or 0)
		comments = int(comments or 0)
		if views > 0:
			engagement_rate = (likes + comments * 5) / views
			return engagement_rate * 1000.0
		return 0.0
	except Exception:
		return 0.0


def parse_duration(iso8601_duration: str) -> float:
	"""Convert an ISO8601 duration (e.g., PT7M32S) to minutes as float."""
	if not iso8601_duration:
		return 0.0

	pattern = r"PT(?:(\d+)H)?(?:(\d+)M)?(?:(\d+)S)?"
	match = re.match(pattern, iso8601_duration)
	if not match:
		return 0.0

	hours = int(match.group(1) or 0)
	minutes = int(match.group(2) or 0)
	seconds = int(match.group(3) or 0)

	total_minutes = hours * 60 + minutes + seconds / 60.0
	return float(total_minutes)


