FROM python:3.11-slim

WORKDIR /app

# Dépendances système requises par OpenMP (LightGBM)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Ports exposés : 8000 pour l'API d'inférence, 8501 pour Streamlit
EXPOSE 8000 8501

# Lancement simultané du serveur API et de l'interface utilisateur
CMD sh -c "uvicorn api:app --host 0.0.0.0 --port 8000 & streamlit run app.py --server.port 8501 --server.address 0.0.0.0"