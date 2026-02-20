import re
import math
from urllib.parse import urlparse
from collections import Counter

def normalize_url(url: str) -> str:
    """Strip scheme to match training data format"""
    url = url.strip()
    if url.startswith("https://"):
        url = url[8:]
    elif url.startswith("http://"):
        url = url[7:]
    return url


def calculate_entropy(string):
    if not string:
        return 0
    counter = Counter(string)
    length = len(string)
    entropy = -sum(
        (count / length) * math.log2(count / length)
        for count in counter.values()
    )
    return round(entropy, 4)


def extract_features(url: str) -> dict:
    url = normalize_url(url)
    if url.startswith("https://"):
        url = url[8:]
    elif url.startswith("http://"):
        url = url[7:]
    try:
        parsed = urlparse("http://" + url)
        domain = parsed.netloc
        path = parsed.path

        features = {
            # Length based
            "url_length": len(url),
            "domain_length": len(domain),
            "path_length": len(path),

            # Count based
            "dot_count": url.count("."),
            "hyphen_count": url.count("-"),
            "underscore_count": url.count("_"),
            "slash_count": url.count("/"),
            "question_count": url.count("?"),
            "equal_count": url.count("="),
            "at_count": url.count("@"),
            "percent_count": url.count("%"),
            "digit_count": sum(c.isdigit() for c in url),

            # Binary flags
            "has_https": 1 if parsed.scheme == "https" else 0,
            "has_ip": 1 if re.match(
                r"(\d{1,3}\.){3}\d{1,3}", domain
            ) else 0,
            "has_at_symbol": 1 if "@" in url else 0,
            "has_double_slash": 1 if "//" in path else 0,
            "has_prefix_suffix": 1 if "-" in domain else 0,

            # Entropy
            "domain_entropy": calculate_entropy(domain),
            "url_entropy": calculate_entropy(url),

            # Subdomain count
            "subdomain_count": max(
                len(domain.split(".")) - 2, 0
            ) if domain else 0,

            # URL depth
            "url_depth": len(
                [p for p in path.split("/") if p]
            ),
        }

        return features

    except Exception:
        return {key: 0 for key in [
            "url_length", "domain_length", "path_length",
            "dot_count", "hyphen_count", "underscore_count",
            "slash_count", "question_count", "equal_count",
            "at_count", "percent_count", "digit_count",
            "has_https", "has_ip", "has_at_symbol",
            "has_double_slash", "has_prefix_suffix",
            "domain_entropy", "url_entropy",
            "subdomain_count", "url_depth"
        ]}


def features_to_list(features: dict) -> list:
    """Converts feature dict to ordered list for ML model input"""
    return list(features.values())