import os
import re
import requests
from dotenv import load_dotenv
from datetime import datetime, timedelta

load_dotenv()

API_KEY = os.getenv("NEWSDATA_API_KEY")
BASE_URL = "https://newsdata.io/api/1/news"

_cache = {}
_cache_timestamps = {}
API_CALL_COUNT = 0
MAX_API_CALLS = 30

STOPWORDS = {
    'a', 'an', 'the', 'is', 'was', 'are', 'were', 'in', 'on', 'at', 'to', 'for',
    'of', 'by', 'with', 'and', 'or', 'but', 'has', 'have', 'had', 'be', 'been',
    'its', 'it', 'this', 'that', 'from', 'as', 'will', 'would', 'can', 'could',
    'may', 'might', 'shall', 'should', 'does', 'do', 'did', 'not', 'no',
    'says', 'said', 'new', 'after', 'before', 'into', 'over', 'more', 'also',
    'very', 'just', 'than', 'then', 'now', 'about', 'up', 'down', 'out',
    'get', 'got', 'make', 'made', 'take', 'took', 'us', 'uk',
}

def _can_call_api():
    global API_CALL_COUNT
    return API_CALL_COUNT < MAX_API_CALLS

def _increment_api_call():
    global API_CALL_COUNT
    API_CALL_COUNT += 1

def get_api_call_count():
    return API_CALL_COUNT

def _extract_keywords(text, max_words=5):
    words = re.findall(
        r'\b[A-Z][a-z]+\b|\b[A-Z]{2,}\b|\b\d+[\.\d%]*\b|\b[a-z]{4,}\b', text
    )
    keywords = [w for w in words if w.lower() not in STOPWORDS and len(w) > 1]
    seen = set()
    unique = []
    for w in keywords:
        if w.lower() not in seen:
            seen.add(w.lower())
            unique.append(w)
    return ' '.join(unique[:max_words])

def _extract_core_subject(text):
    proper = re.findall(
        r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b|\b[A-Z]{2,}\b', text
    )
    filtered = [
        pn for pn in proper
        if len(pn) > 2 and pn.lower() not in STOPWORDS
    ]
    seen = set()
    unique = []
    for w in filtered:
        w_clean = w.strip()
        if w_clean.lower() not in seen:
            seen.add(w_clean.lower())
            unique.append(w_clean)
    result = ' '.join(unique[:3])
    return result if len(unique) >= 2 else ''

def _extract_short_query(text, max_words=3):
    proper = re.findall(r'\b[A-Z][a-z]+\b|\b[A-Z]{2,}\b', text)
    lower = re.findall(r'\b[a-z]{4,}\b', text)
    combined = [w for w in proper if w.lower() not in STOPWORDS]
    combined += [w for w in lower if w.lower() not in STOPWORDS]
    seen = set()
    unique = []
    for w in combined:
        wl = w.lower()
        if wl not in seen:
            seen.add(wl)
            unique.append(w)
    result = ' '.join(unique[:max_words])
    return result if len(result) > 4 else ''

def _search_api(query, country="in", language="en", size=5):
    cache_key = f"search:{query}:{country}:{size}"
    if cache_key in _cache:
        cached_time = _cache_timestamps.get(cache_key)
        if cached_time and datetime.now() - cached_time < timedelta(hours=1):
            return _cache[cache_key]

    if not _can_call_api():
        return {"error": "API credit limit reached", "results": []}

    if not API_KEY or API_KEY.strip() == "" or API_KEY == "your_api_key_here":
        return {"error": "API key not configured", "results": []}

    try:
        _increment_api_call()
        resp = requests.get(
            BASE_URL,
            params={
                "apikey": API_KEY,
                "q": query,
                "country": country,
                "language": language,
                "size": size,
            },
            timeout=10,
        )
        data = resp.json()
        if data.get("status") == "success":
            results = data.get("results", [])
            _cache[cache_key] = {"results": results, "source": "api"}
            _cache_timestamps[cache_key] = datetime.now()
            return {"results": results, "source": "api"}
        return {"error": data.get("results", "API error"), "results": []}
    except Exception as e:
        return {"error": str(e), "results": []}

def smart_search_news(headline, country="in", language="en"):
    keywords = _extract_keywords(headline, max_words=5)
    core = _extract_core_subject(headline)
    short = _extract_short_query(headline, max_words=3)

    queries = []
    if short and short != keywords and short != core:
        queries.append(("short", short))
    if keywords and len(keywords) > 3 and keywords != short:
        queries.append(("keywords", keywords))
    if core and core != keywords and core != short:
        queries.append(("core", core))
    queries.append(("full", headline))

    attempts = 0
    for label, query in queries:
        if attempts >= 2:
            break
        if not query or len(query) < 4:
            continue
        attempts += 1
        result = _search_api(query, country, language)
        if result.get("results"):
            result["query_used"] = query
            result["query_label"] = label
            return result

    last = queries[-1][1] if queries else headline
    result = _search_api(last, country, language)
    result["query_used"] = last
    result["query_label"] = "fallback"
    return result

def search_news(query, country="in", language="en"):
    return _search_api(query, country, language)

def get_latest_news(country="in", category="top", size=10):
    cache_key = f"latest:{country}:{category}"
    if cache_key in _cache:
        cached_time = _cache_timestamps.get(cache_key)
        if cached_time and datetime.now() - cached_time < timedelta(minutes=30):
            return _cache[cache_key]

    if not _can_call_api():
        return {"error": "API credit limit reached", "results": []}

    if not API_KEY or API_KEY.strip() == "" or API_KEY == "your_api_key_here":
        return {"error": "API key not configured", "results": []}

    try:
        _increment_api_call()
        params = {"apikey": API_KEY, "country": country, "language": "en", "size": size}
        if category != "top":
            params["category"] = category
        resp = requests.get(BASE_URL, params=params, timeout=10)
        data = resp.json()
        if data.get("status") == "success":
            results = data.get("results", [])
            _cache[cache_key] = {"results": results, "source": "api"}
            _cache_timestamps[cache_key] = datetime.now()
            return {"results": results, "source": "api"}
        return {"error": data.get("results", "API error"), "results": []}
    except Exception as e:
        return {"error": str(e), "results": []}
