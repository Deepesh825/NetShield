# 🛡️ NetShield - URL Safety Intelligence
### AI-Powered Multi-Layer URL Threat Detection System

NetShield is a fast, resource-efficient threat intelligence engine that analyzes URLs to detect phishing, malware, and suspicious domains in real-time. Built specifically to bypass the limitations of standard serverless deployments.

NetShield is a full-stack security tool that analyzes URLs and determines whether they are **Safe** or **Malicious** using a multi-layer detection pipeline combining:

- 🌐 DNS validation
- 📅 Domain age checks
- 🤖 Machine Learning model
- 🔬 VirusTotal verification

Built using **FastAPI + Scikit-Learn + Static Frontend**.

---

## 🔥 Features

- ✅ Real-time URL threat detection
- 🧠 ML-based phishing detection
- 🔎 VirusTotal fallback verification
- 📊 Confidence scoring & risk factors
- 🕒 Session tracking history
- 🌙 Light / Dark / System theme UI
- ⚡ Clean, responsive UI
- 🚀 Deployment-ready (Render compatible)

---

## 🏗️ Architecture
Instead of relying solely on expensive external APIs or a single Machine Learning model, NetShield cashandlescades traffic through three optimized security layers:

```
Browser
  │
  ▼
FastAPI Backend
  │
  ├── Tier 0 → DNS & Domain Age Check
  ├── Tier 1 → ML Model Prediction
  └── Tier 2 → VirusTotal API (In case of low confidence of ML)
  │
  ▼
Final Verdict (Safe / Malicious)
```

---

## 📁 Project Structure

```
NetShield/
│
├── backend/
│   ├── core/
│   │   ├── analyzer.py
│   │   ├── dns_checker.py
│   │   ├── feature_extractor.py
│   │   ├── ml_model.py
│   │   └── virustotal.py
│   │
│   ├── model/
│   │   ├── train.py
│   │   ├── url_model.pkl
│   │   ├── malicious_phish.csv
│   │   └── tranco_top10k.csv
│   │
│   ├── main.py
│   └── .env
│
├── frontend/
│   └── index.html
│
├── requirements.txt
└── README.md
```

---

## 🧠 How It Works

### 🟢 Tier 0 – DNS Validation
Checks if the domain actually exists on the internet. If it doesn't resolve, it's immediately flagged as malicious.

### 🟡 Tier 1 – AI Model
Extracts 26+ engineered features (also known as lexical features) from the URL:

- Length
- Special characters
- Suspicious keywords
- Domain structure
- Pattern anomalies

Model output:
```
(Safe / Malicious) + Confidence %
```

### 🔴 Tier 2 – VirusTotal (External API)
If the ML model confidence is low or inconclusive, the URL is cross-verified against:

- 70+ antivirus engines
- Global threat intelligence databases

---

## ⚙️ Tech Stack

| Layer | Technology |
|-------|-----------|
| Backend | FastAPI |
| ML | Scikit-Learn |
| Data | Pandas |
| Model Storage | Git LFS |
| Frontend | HTML + CSS + Vanilla JS |
| Deployment | Render |
| External API | VirusTotal |

---

## 🚀 Running Locally (for contribution or testing)

### 1. Clone Repository

```bash
git clone https://github.com/Deepesh825/NetShield.git
cd NetShield
```

### 2. Create Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
source venv/bin/activate   # macOS / Linux
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Insert your unique api key (from [virustotal's website](https://www.virustotal.com/)) inside the `.env` file.

```bash
VIRUSTOTAL_API_KEY=your_api_key_here
```

### 4.  Run Server

```bash
uvicorn backend.main:app --reload
```

Then visit: `http://127.0.0.1:8000`

---

## 🌍 Website
This project is deployed via Render.  
Link [here](https://netshield-lu9j.onrender.com/)


---

## 📊 Model Details

| Property | Detail |
|----------|--------|
| Dataset | Phishing URLs + Tranco Top Domains |
| File Size | ~250MB (stored via Git LFS) |
| Algorithm | Random Forest Classifier |
| Features | 26+ engineered URL features |

---

## 🔐 Security Disclaimer

This tool provides automated analysis but should **not** replace professional cybersecurity solutions. Always use multiple sources when evaluating link safety :)

---

## 👨‍💻 Team
This project was made successfull with the collaborative efforts of our team.

**Deepesh Goyal**  
BTech CSE Student | Backend and frontend  
**Shaurya Pethe**  
BTech CSE (AIML) Student | Backend pipelines and ML  
**Pratik Kumawat**  
BTech CSE Student | Backend and ML research  
**Archie Ayush**  
BTech CSE Student | Frontend Design  



---

## ⭐ Future Improvements

- [ ] Lightweight model optimization
- [ ] Docker support
- [ ] Threat explanation visualizer
- [ ] Rate limiting
- [ ] Authentication layer
- [ ] Admin dashboard
- [ ] Web browser extension

---

## Contributions  
Contributions and suggestions are most welcome. We'd love to see new ideas to make our project even better!

*If you found this project useful, give it a ⭐ :)*