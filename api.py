from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import pandas as pd
import sqlite3
import os
import shutil
from datetime import datetime

app = FastAPI(
    title="Dakar Immo AI - API d'Inférence MLOps",
    description="API REST de prédiction des loyers à Dakar avec journalisation SQLite et gestion de Rollback.",
    version="1.0.0"
)

# Registre des versions pour la procédure de Rollback
MODEL_REGISTRY = {
    "v1.0_champion": "models/model_dakar_immo.pkl",
    "v0.9_backup": "models/model_dakar_immo_backup.pkl"
}
ACTIVE_VERSION = "v1.0_champion"

# Création d'une copie de sauvegarde locale pour valider la procédure de Rollback
if os.path.exists("models/model_dakar_immo.pkl") and not os.path.exists("models/model_dakar_immo_backup.pkl"):
    shutil.copy("models/model_dakar_immo.pkl", "models/model_dakar_immo_backup.pkl")

model = joblib.load(MODEL_REGISTRY[ACTIVE_VERSION])

# Schéma de validation des requêtes entrantes
class PropertyInput(BaseModel):
    quartier: str = "Mermoz"
    nb_pieces: int = 3
    nb_salles_bain: int = 2
    dist_plateau_km: float = 5.2
    dist_vdn_km: float = 1.5
    dist_mer_km: float = 1.2
    is_meuble: int = 1
    has_piscine: int = 0
    has_parking: int = 1
    has_ascenseur: int = 1
    has_securite: int = 1
    has_groupe: int = 1
    has_vue_mer: int = 0

@app.get("/health", tags=["Monitoring & Santé"])
def health_check():
    """Contrôle de santé du service et vérification de la version active."""
    return {
        "status": "healthy",
        "active_model_version": ACTIVE_VERSION,
        "champion_mae_fcfa": 164107.9,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict", tags=["Inférence ML"])
def predict_rent(prop: PropertyInput):
    """Calcule le loyer estimé et journalise la requête pour le suivi de Data Drift."""
    ratio = round(prop.nb_salles_bain / max(1, prop.nb_pieces), 2)
    
    df_in = pd.DataFrame([{
        'nb_pieces': prop.nb_pieces,
        'nb_salles_bain': prop.nb_salles_bain,
        'ratio_bains_chambres': ratio,
        'dist_plateau_km': prop.dist_plateau_km,
        'dist_vdn_km': prop.dist_vdn_km,
        'dist_mer_km': prop.dist_mer_km,
        'has_groupe': prop.has_groupe,
        'has_reserve_eau': 1 if prop.has_piscine else 0,
        'is_studio': 1 if prop.nb_pieces == 1 else 0,
        'is_villa': 1 if prop.nb_pieces >= 5 else 0,
        'has_piscine': prop.has_piscine,
        'has_ascenseur': prop.has_ascenseur,
        'has_parking': prop.has_parking,
        'has_securite': prop.has_securite,
        'has_clim': 1 if prop.is_meuble else 0,
        'has_vue_mer': prop.has_vue_mer,
        'is_meuble': prop.is_meuble,
        'has_cuisine_equipee': prop.is_meuble,
        'has_balcon': 1,
        'is_agence_pro': 1,
        'longueur_description': 120,
        'quartier': prop.quartier,
        'description_complete': f"Appartement {prop.nb_pieces} pièces à louer à {prop.quartier}"
    }])
    
    pred_fcfa = int(model.predict(df_in)[0])
    
    # Journalisation dans la table SQLite 'predictions_logs'
    db_file = "data/dakar_immo.db" if os.path.exists("data/dakar_immo.db") else "db/dakar_immo.db"
    if os.path.exists(db_file):
        conn = sqlite3.connect(db_file)
        conn.execute("""
            INSERT INTO predictions_logs (timestamp, quartier, nb_pieces, nb_salles_bain, is_meuble, has_piscine, loyer_predit_fcfa, model_version)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            prop.quartier,
            prop.nb_pieces,
            prop.nb_salles_bain,
            prop.is_meuble,
            prop.has_piscine,
            pred_fcfa,
            ACTIVE_VERSION
        ))
        conn.commit()
        conn.close()

    return {
        "quartier": prop.quartier,
        "loyer_estime_fcfa": pred_fcfa,
        "fourchette_marche_fcfa": [max(30000, pred_fcfa - 164107), pred_fcfa + 164107],
        "model_version": ACTIVE_VERSION
    }

@app.post("/rollback/{target_version}", tags=["Administration & Rollback"])
def rollback_model(target_version: str):
    """Bascule à chaud vers une version précédente du modèle sans interruption de service."""
    global ACTIVE_VERSION, model
    if target_version not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail="Version cible absente du registre.")
    
    ACTIVE_VERSION = target_version
    model = joblib.load(MODEL_REGISTRY[ACTIVE_VERSION])
    return {
        "status": "success",
        "message": f"Rollback effectué avec succès vers la version : {ACTIVE_VERSION}"
    }