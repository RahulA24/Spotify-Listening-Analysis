# Spotify AI Engineer: Data Agent and Listening Analysis

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat&logo=python)
![Scikit-Learn](https://img.shields.io/badge/ML-Random%20Forest-orange?style=flat&logo=scikit-learn)
![Agentic Workflow](https://img.shields.io/badge/AI-Agentic%20Workflow-green?style=flat)
![Status](https://img.shields.io/badge/Status-Production%20Ready-success)

## Project Overview
This project is an end to end AI engineering pipeline built around Spotify streaming history.  
It goes beyond simple charts and reports by combining a **robust ETL workflow**, a **skip-prediction model**, and an **interactive data agent** that lets you query your listening habits using natural language.

The system is designed to handle both Standard and Extended Spotify exports, automatically inferring schema differences and applying the required transformations. Feature engineering adjusts dynamically based on whatever metadata is available.

---

## Technical Architecture

### 1. Advanced Data Agent (Agentic Workflow)
* **Module:** `agent/chat_bot.py`
* **What it does:** Uses regex-based entity extraction to interpret time-based questions like:
  * *"When do I listen on Sat, Nov 2024?"*
  * *"Who was my top artist in 2023?"*
  * *"Time listened in October 2022"*
* **How it works:** Parses the user query, extracts dates, applies filters, and routes the request to the correct Pandas operation.

### 2. Predictive Modeling (Random Forest)
* **Module:** `src/predictive_model.py`
* **Goal:** Predict whether a track is skipped based on contextual cues.
* **Highlights:**
  * Automatically adds Extended History fields (shuffle mode, start reason) when found.
  * Intentionally excludes `ms_played` to avoid classic duration-based leakage.
  * Focuses on behavioral signals like time of day, weekday patterns, and weekend usage.

### 3. Resilient ETL and Feature Engineering
* **Modules:** `src/etl.py`, `src/feature_eng.py`
* **Behavior:**
  * Detects whether input is Standard or Extended Spotify history.
  * Renames, normalizes, or preserves columns accordingly.
  * Filters out micro-plays of only a few seconds unless the user explicitly skipped the track.
* **Output:** A clean, unified DataFrame ready for modeling or querying.

### 4. Unsupervised Learning
* **Module:** `src/clustering.py`
* **Approach:** K-Means clustering on engineered artist metrics to group them into usage-based segments such as frequently played favorites or high-skip artists.

---

## Project Structure

```bash
Spotify-Listening-Analysis/
├── agent/                  # Data Agent logic
│   ├── chat_bot.py
├── data/
│   ├── raw/                # Spotify JSON files go here
├── src/
│   ├── etl.py              # Schema-aware loader
│   ├── feature_eng.py      # Feature engineering utilities
│   ├── predictive_model.py # Random Forest training
│   ├── clustering.py       # Artist segmentation
├── main.py                 # CLI entry point
└── README.md


##How to Run Locally
1. Clone the repository
Bash

git clone [https://github.com/RahulA24/Spotify-listening-Analysis.git](https://github.com/RahulA24/Spotify-listening-Analysis.git)
cd Spotify-listening-Analysis

2. Install dependencies

pip install pandas scikit-learn matplotlib seaborn

3. Add Data
Place your StreamingHistory.json (or the new Audio_History.json) files inside the data/raw/ folder. The pipeline supports multiple files and mixed formats.

4. Run the Application
Execute the main entry point to run the ETL, train the models, and launch the Interactive Agent:

python main.py

##Sample Agent InteractionPlaintext

SPOTIFY DATA AGENT IS LIVE!
Data Coverage: January 2020 to November 2024

YOU: When do I listen on Sat, Nov 2024?
AGENT: Peak Listening Time (Saturdays Nov 2024): Around 2 PM PM.

YOU: Who is my top artist in 2023?
AGENT: Top Artist (2023): The Weeknd (452 plays).


##Author: Rahul, AI Engineer / Data Engineer
