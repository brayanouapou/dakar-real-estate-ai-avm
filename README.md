# 🏠 Dakar Immo AI — Intelligence Artificielle & Modélisation Foncière à Dakar

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![Framework](https://img.shields.io/badge/Streamlit-1.30+-red.svg)](https://streamlit.io/)
[![API](https://img.shields.io/badge/FastAPI-0.100+-green.svg)](https://fastapi.tiangolo.com/)
[![ML Model](https://img.shields.io/badge/Champion_Model-LightGBM-orange.svg)](https://lightgbm.readthedocs.io/)
[![MLOps](https://img.shields.io/badge/Tracking-MLflow-blueviolet.svg)](https://mlflow.org/)
[![License](https://img.shields.io/badge/License-Academic-lightgrey.svg)]()

> **Projet de Mémoire MBA Big Data & Intelligence Artificielle — Institut Supérieur de Management (ISM) Dakar**  
> *Développement d’un système d'évaluation automatisé (Automated Valuation Model - AVM) pour le marché immobilier locatif informel et polycentrique de la région de Dakar.*

---

## 📌 Présentation du Projet

Le marché locatif dakarois est caractérisé par une forte opacité informationnelle, des négociations de gré à gré et une absence quasi systématique de la surface habitable exacte ($m^2$) dans les annonces en ligne.

**Dakar Immo AI** apporte une solution basée sur les données pour réguler et estimer la valeur locative des biens en combinant :
- Une collecte multi-sources à grande échelle (CoinAfrique, Expat-Dakar, WhatsApp, Facebook Marketplace).
- Un pipeline NLP pour l'extraction regex des caractéristiques structurelles et de standing.
- Des variables géospatiales captant la polycentricité de la métropole dakaroise (Plateau, VDN, façade maritime)[cite: 8].
- Une modélisation supervisée avec sélection d'un **modèle champion LightGBM** ($R^2 = 0.6400$)[cite: 8].
- Une suite MLOps complète : API d'inférence, journalisation SQLite, suivi de modèle sous MLflow et surveillance du *Data Drift*[cite: 8].

---

## 🏗️ Architecture Technique

```mermaid
flowchart LR
    A[Sources Web & Messageries<br/>Bronze] -->|Extraction Regex & NLP| B[Zone Silver<br/>3 838 annonces]
    B -->|Calculs Haversine & Imputation| C[Zone Gold<br/>3 515 annonces]
    C -->|Dédoublonnage Anti-Leakage| D[Dataset Train<br/>2 905 annonces]
    D -->|Entraînement & Benchmark| E[Modèle Champion<br/>LightGBM .pkl]
    E --> F[FastAPI REST API]
    E --> G[Streamlit UI]
    F -->|Logs Requêtes| H[(SQLite DB)]
    H -->|Test Kolmogorov-Smirnov| I[Drift Monitor]
