import os
import requests
import base64
from dotenv import load_dotenv

load_dotenv()

def check_virustotal(url: str) -> dict:
    """
    Checks a URL against VirusTotal's v3 API by retrieving its existing report.
    """
    api_key = os.getenv("VIRUS_TOTAL_API_KEY")
    if not api_key:
        return {"error": "VirusTotal API key is missing. Check your .env file."}

    # VirusTotal v3 requires the URL to be a base64 encoded string
    url_id = base64.urlsafe_b64encode(url.encode()).decode().strip("=")
    api_endpoint = f"https://www.virustotal.com/api/v3/urls/{url_id}"
    
    headers = {
        "accept": "application/json",
        "x-apikey": api_key
    }

    try:
        response = requests.get(api_endpoint, headers=headers)
        
        if response.status_code == 200:
            data = response.json().get("data", {}).get("attributes", {})
            stats = data.get("last_analysis_stats", {})
            
            # Determine status based on malicious flags
            malicious_votes = stats.get("malicious", 0)
            status = "Malicious" if malicious_votes > 0 else "Safe"
            
            return {
                "source": "VirusTotal API",
                "prediction": status,
                "malicious_flags": malicious_votes,
                "total_scans": sum(stats.values())
            }
        elif response.status_code == 404:
            return {"source": "VirusTotal API", "prediction": "Unknown", "message": "No historical data found for this URL."}
        else:
            return {"error": f"API returned status code {response.status_code}"}
            
    except requests.exceptions.RequestException as e:
        return {"error": str(e)}