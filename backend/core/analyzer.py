import argparse
import json

from feature_extractor import extract_features
from ml_model import predict_url_full
from virustotal import check_virustotal


def analyze_url(url: str) -> dict:
    """
    NetShield Tiered Security Pipeline
    Tier 1 → Local ML
    Tier 2 → VirusTotal (only when needed)
    """

    
    features = extract_features(url)

    
    prediction, confidence, reasons = predict_url_full(url, features)

    

    confidence_percent = round(confidence * 100, 2)

    # HIGH CONFIDENCE : Skip API
    if confidence >= 0.90 or confidence <= 0.10:
        return {
            "url": url,
            "final_prediction": prediction,
            "confidence": confidence_percent,
            "source": "Local ML (Tier 1)",
            "risk_factors": reasons,
            "details": "High confidence prediction. API skipped."
        }

    

    vt_result = check_virustotal(url)

    # If VirusTotal fails : fallback to ML
    if not vt_result or "error" in vt_result:
        return {
            "url": url,
            "final_prediction": prediction,
            "confidence": confidence_percent,
            "source": "Local ML (Fallback)",
            "risk_factors": reasons,
            "details": "VirusTotal unavailable. Using ML result."
        }

    # If VirusTotal has no data
    if vt_result.get("prediction") == "Unknown":
        return {
            "url": url,
            "final_prediction": prediction,
            "confidence": confidence_percent,
            "source": "Local ML (No VT Data)",
            "risk_factors": reasons,
            "details": "No reputation data found."
        }


    return {
        "url": url,
        "final_prediction": vt_result["prediction"],
        "confidence": "External Verification",
        "source": "VirusTotal (Tier 2)",
        "risk_factors": reasons,
        "details": f"{vt_result.get('malicious_flags', 0)} vendors flagged this URL."
    }



if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="NetShield URL Analyzer - CLI Testing"
    )
    parser.add_argument(
        "-u",
        "--url",
        required=True,
        help="The URL string you want to analyze"
    )

    args = parser.parse_args()

    print(f"\n[NetShield] Analyzing URL: {args.url}")
    print("-" * 60)

    result = analyze_url(args.url)

    print(json.dumps(result, indent=4))
    print("-" * 60)