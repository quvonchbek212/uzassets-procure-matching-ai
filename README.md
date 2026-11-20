# 🎯 UzAssets · Product Limit Article Matching AI Engine

[![Python](https://img.shields.io/badge/Python-3.11%2B-blue.svg)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110%2B-green.svg)](https://fastapi.tiangolo.com)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.4%2B-orange.svg)](https://scikit-learn.org)
[![Tests](https://img.shields.io/badge/Tests-73%20Passed-brightgreen.svg)](tests/)
[![Docker](https://img.shields.io/badge/Deployment-Docker-2496ED.svg)](https://docker.com)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An enterprise-grade procurement classification microservice developed for the **UzAssets Investment Company** portfolio. Automatically maps hundreds of thousands of raw corporate procurement entries and purchase requisitions to standardized budgetary expenditure lines (**Limit Articles**) using multi-tier exact matching, fuzzy phonetic search, and semantic classification.

---

## 🏢 Business Context & Problem

Large enterprise holding companies process hundreds of thousands of distinct procurement line items annually. Manually mapping diverse, messy item descriptions (*e.g., "M-10G2k motor oil 200L" vs "Motor moyi 200 litr"*) to formal corporate budget articles causes:
1. Significant delays in budget allocation and financial approval.
2. High human error rates in expense classification.
3. Lack of cross-subsidiary procurement visibility.

This engine automates the entire classification pipeline with **sub-millisecond latency** and high precision.

---

## ⚡ Multi-Tier Classification Pipeline

```text
[Incoming Item Query]
         │
         ▼
┌─────────────────────────────────┐
│ Tier 1: Canonical Hash Lookup   │ ──(Match Found: Score = 1.0)──► [Instant Output (<2ms)]
└────────────────┬────────────────┘
                 │ (No exact match)
                 ▼
┌─────────────────────────────────┐
│ Tier 2: Fuzzy Phonetic Matching │ ──(Similarity >= Threshold)───► [High-Confidence Output]
│ (Levenshtein + Token Sort)      │
└────────────────┬────────────────┘
                 │ (Ambiguous match)
                 ▼
┌─────────────────────────────────┐
│ Tier 3: TF-IDF & ML Classifier  │ ──(Probabilistic Match)───────► [Calibrated Prediction]
└─────────────────────────────────┘
```

* **Tier 1 (Canonical Dictionary):** O(1) hash table lookup for verified historical purchase descriptions.
* **Tier 2 (Fuzzy Token Matching):** Normalized token-ratio string similarity resilient against typos, morphological suffixes, and transliteration differences (Uzbek Latin / Cyrillic / Russian).
* **Tier 3 (Supervised Classification):** TF-IDF vectorization with regularized linear/ensemble classification as a robust fallback.

---

## 🧪 Rigorous Verification & Testing

The service includes an exhaustive automated test suite covering:
* Exact keyword and boundary matching.
* Robustness against spelling mistakes and bilingual terminology.
* Edge cases: empty descriptions, special symbols, multi-line inputs.
* Concurrency and latency under high load.

```bash
# Run the automated test suite
pytest tests/ -v
# Output: 73 passed in 1.42s
```

---

## 🚀 API Endpoints

### Single Item Prediction
`POST /api/v1/match`
```json
{
  "product_name": "Mobil Super 3000 5W-40 motor moyi 4L",
  "unit_of_measure": "litr",
  "category_hint": "Yoqilg'i-moylash"
}
```
**Response:**
```json
{
  "limit_article_code": "ART-2041",
  "limit_article_name": "Yoqilg'i va moylash materiallari (YMM)",
  "confidence_score": 0.985,
  "match_tier": "tier_1_exact_canonical"
}
```

### High-Throughput Batch Processing
`POST /api/v1/batch-match`
Enables parallel batch classification of up to 10,000 procurement lines per request for enterprise ERP bulk imports.

---

## 🛠 Quickstart

```bash
# Clone repository
git clone https://github.com/quvonchbek212/uzassets-procure-matching-ai.git
cd uzassets-procure-matching-ai

# Install dependencies
python -m venv .venv
source .venv/bin/activate  # Or on Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Run server
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

---

## 📄 License
This project is licensed under the [MIT License](LICENSE).
