import sqlite3
import pandas as pd
import os
from datetime import datetime

DB_PATH = "db/dakar_immo.db"
CSV_PATH = "data/processed/dataset_dakar_gold.csv"

os.makedirs("db", exist_ok=True)
conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# 1. Import du Dataset Gold dans la table SQL 'annonces_gold'
df = pd.read_csv(CSV_PATH, sep=';')
df.columns = df.columns.str.strip().str.lower()
cols_bizarres = [c for c in df.columns if ',' in c]
if cols_bizarres:
    df = df.drop(columns=cols_bizarres)

df.to_sql("annonces_gold", conn, if_exists="replace", index=False)

# 2. Création de la table de monitoring 'predictions_logs' pour la production
cursor.execute("""
CREATE TABLE IF NOT EXISTS predictions_logs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT,
    quartier TEXT,
    nb_pieces INTEGER,
    nb_salles_bain INTEGER,
    is_meuble INTEGER,
    has_piscine INTEGER,
    loyer_predit_fcfa INTEGER,
    model_version TEXT
)
""")

# Insertion de quelques requêtes de test pour alimenter le monitoring
sample_logs = [
    (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Almadies", 4, 3, 1, 1, 1250000, "LightGBM_v1.0"),
    (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Mermoz", 3, 2, 0, 0, 680000, "LightGBM_v1.0"),
    (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Ouakam", 2, 1, 1, 0, 320000, "LightGBM_v1.0"),
    (datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "Sacré-Cœur", 3, 2, 0, 0, 490000, "LightGBM_v1.0"),
]
cursor.executemany("""
INSERT INTO predictions_logs (timestamp, quartier, nb_pieces, nb_salles_bain, is_meuble, has_piscine, loyer_predit_fcfa, model_version)
VALUES (?, ?, ?, ?, ?, ?, ?, ?)
""", sample_logs)

conn.commit()

# Affichage pour la capture d'écran dans le terminal
print("✅ Base de données SQLite initialisée avec succès : data/dakar_immo.db")
res = cursor.execute("SELECT COUNT(*) FROM annonces_gold").fetchone()[0]
print(f"📊 Table 'annonces_gold' : {res} lignes enregistrées.")
print("\n🔍 Aperçu de la table de production 'predictions_logs' :")
df_logs = pd.read_sql("SELECT * FROM predictions_logs", conn)
print(df_logs.to_string(index=False))

conn.close()