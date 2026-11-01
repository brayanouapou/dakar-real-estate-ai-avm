import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import folium
from streamlit_folium import st_folium
import plotly.express as px

# =====================================================================
# 1. CONFIGURATION ET STYLES CSS
# =====================================================================
st.set_page_config(
    page_title="Dakar Immo AI - Intelligence Immobilière", 
    page_icon="🏠", 
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main-title { font-size: 32px; color: #1E3A8A; font-weight: bold; margin-bottom: 0px; }
    .sub-title { font-size: 16px; color: #4B5563; margin-bottom: 20px; }
    .metric-card {
        background-color: #F3F4F6;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #1E3A8A;
        margin-bottom: 15px;
    }
    .stTabs [data-baseweb="tab-list"] { gap: 10px; }
    .stTabs [data-baseweb="tab"] {
        padding-top: 10px;
        padding-bottom: 10px;
        font-weight: 600;
    }
    </style>
""", unsafe_allow_html=True)

# =====================================================================
# 2. CHARGEMENT ROBUSTE DU MODÈLE ET DES DONNÉES GOLD
# =====================================================================
@st.cache_resource
def load_ml_model():
    model_path = 'models/model_dakar_immo.pkl'
    if not os.path.exists(model_path):
        st.error(f"⚠️ Le fichier modèle '{model_path}' est introuvable.")
        st.stop()
    return joblib.load(model_path)

@st.cache_data
def load_gold_data():
    data_path = 'data/processed/dataset_dakar_gold.csv'
    if not os.path.exists(data_path):
        data_path = 'dataset_dakar_gold.csv'
    
    df = pd.read_csv(data_path, sep=';')
    df.columns = df.columns.str.strip().str.lower()
    df = df[df['nb_salles_bain'] <= 10].copy()
    if 'quartier' in df.columns:
        df['quartier'] = df['quartier'].fillna('Inconnu')
    return df

try:
    model_pipeline = load_ml_model()
    df_gold = load_gold_data()
except Exception as e:
    st.error(f"⚠️ Erreur de chargement des composants ML : {e}")
    st.stop()

# Coordonnées et métriques par quartier
quartiers_coords = df_gold.groupby('quartier')[['latitude', 'longitude']].mean().to_dict(orient='index')
quartiers_medians = df_gold.groupby('quartier')['prix_fcfa'].median().to_dict()
quartiers_liste = sorted([q for q in df_gold['quartier'].unique() if q != 'Inconnu'])

POINTS_INTERET = {
    'plateau': (14.668, -17.435),
    'vdn': (14.720, -17.465),
    'mer': (14.745, -17.518)
}

def distance_haversine(lat1, lon1, lat2, lon2):
    from math import radians, cos, sin, asin, sqrt
    if pd.isna(lat1) or pd.isna(lon1): return 5.0
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1) * cos(lat2) * sin(dlon/2)**2
    return round(2 * R * asin(sqrt(a)), 2)

# =====================================================================
# 3. EN-TÊTE
# =====================================================================
st.markdown('<p class="main-title">🏠 Dakar Immo AI</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Plateforme d\'estimation, de recommandation budgétaire et d\'analyse du marché locatif dakarois</p>', unsafe_allow_html=True)

tab1, tab2, tab3, tab4 = st.tabs([
    "🔮 Estimateur ML", 
    "🎯 Recommandateur par Budget", 
    "🗺️ Carte Interactive du Marché",
    "📢 Guide des Plateformes"
])

# =====================================================================
# ONGLET 1 : ESTIMATEUR DE PRIX (ML CHAMPION LIGHTGBM)
# =====================================================================
with tab1:
    col_input, col_results = st.columns([1.1, 0.9])
    
    with col_input:
        st.subheader("📝 Caractéristiques du Bien")
        
        default_index = quartiers_liste.index("Ouakam") if "Ouakam" in quartiers_liste else 0
        quartier_sel = st.selectbox("Quartier", options=quartiers_liste, index=default_index)
        
        c1, c2 = st.columns(2)
        with c1:
            nb_pieces = st.number_input("Nombre de Pièces", min_value=1, max_value=10, value=3)
            nb_bains = st.number_input("Salles de Bain", min_value=1, max_value=8, value=2)
        with c2:
            is_meuble = st.selectbox("Meublé ?", [0, 1], format_func=lambda x: "Oui" if x==1 else "Non")
            type_bien = st.selectbox("Type de Bien", ["Appartement", "Villa", "Studio"])

        st.write("**Équipements & Standing :**")
        eq1, eq2, eq3 = st.columns(3)
        with eq1:
            has_piscine = st.checkbox("🏊‍♂️ Piscine", value=False)
            has_parking = st.checkbox("🅿️ Parking / Garage", value=False)
        with eq2:
            has_ascenseur = st.checkbox("🛗 Ascenseur", value=False)
            has_securite = st.checkbox("👮‍♂️ Sécurité H24", value=False)
        with eq3:
            has_vue_mer = st.checkbox("🌊 Vue sur Mer", value=False)
            has_groupe = st.checkbox("⚡ Groupe Électrogène", value=False)

    # Calcul des caractéristiques géographiques et dérivées
    lat = quartiers_coords.get(quartier_sel, {}).get('latitude', 14.720)
    lon = quartiers_coords.get(quartier_sel, {}).get('longitude', -17.465)
    
    dist_p = distance_haversine(lat, lon, POINTS_INTERET['plateau'][0], POINTS_INTERET['plateau'][1])
    dist_v = distance_haversine(lat, lon, POINTS_INTERET['vdn'][0], POINTS_INTERET['vdn'][1])
    dist_m = distance_haversine(lat, lon, POINTS_INTERET['mer'][0], POINTS_INTERET['mer'][1])
    
    ratio_bains = round(nb_bains / nb_pieces, 2)
    is_villa = 1 if type_bien == "Villa" else 0
    is_studio = 1 if (type_bien == "Studio" or nb_pieces == 1) else 0

    input_dict = {
        'nb_pieces': [nb_pieces],
        'nb_salles_bain': [nb_bains],
        'ratio_bains_chambres': [ratio_bains],
        'dist_plateau_km': [dist_p],
        'dist_vdn_km': [dist_v],
        'dist_mer_km': [dist_m],
        'has_groupe': [int(has_groupe)],
        'has_reserve_eau': [1 if (has_piscine or is_villa) else 0],
        'is_studio': [is_studio],
        'is_villa': [is_villa],
        'has_piscine': [int(has_piscine)],
        'has_ascenseur': [int(has_ascenseur)],
        'has_parking': [int(has_parking)],
        'has_securite': [int(has_securite)],
        'has_clim': [1 if (is_meuble or has_piscine) else 0],
        'has_vue_mer': [int(has_vue_mer)],
        'is_meuble': [is_meuble],
        'has_cuisine_equipee': [1 if is_meuble else 0],
        'has_balcon': [1],
        'is_agence_pro': [1],
        'longueur_description': [120],
        'quartier': [quartier_sel],
        'description_complete': [f"Appartement {nb_pieces} pièces à louer à {quartier_sel} avec {nb_bains} salles de bain."]
    }
    
    df_input = pd.DataFrame(input_dict)
    
    # Exécution de la prédiction via le Pipeline
    try:
        pred_prix = int(model_pipeline.predict(df_input)[0])
    except Exception as err:
        pred_prix = 400000
        st.warning(f"Calcul simplifié appliqué ({err})")

    with col_results:
        st.subheader("💡 Estimation du Modèle Champion (LightGBM)")
        
        st.metric(label="Loyer Mensuel Estimé", value=f"{pred_prix:,.0f} FCFA".replace(",", " "))
        
        # Intervalle basé sur la MAE réelle du LightGBM (164 107 FCFA)
        mae_lightgbm = 164107
        borne_basse = max(30000, pred_prix - mae_lightgbm)
        borne_haute = pred_prix + mae_lightgbm
        
        st.caption(f"🎯 Fourchette d'incertitude ML (MAE ±164k FCFA) : **{borne_basse:,.0f} FCFA** à **{borne_haute:,.0f} FCFA**".replace(",", " "))
        st.divider()
        
        med_local = quartiers_medians.get(quartier_sel, 400000)
        st.write(f"• Loyer médian observé à **{quartier_sel}** : `{med_local:,.0f} FCFA`".replace(",", " "))
        
        diff_pct = round((pred_prix - med_local) / med_local * 100, 1)
        if diff_pct > 0:
            st.info(f"Ce bien se situe **+{diff_pct}% au-dessus** de la médiane de {quartier_sel} (justifié par le niveau d'équipement/standing).")
        else:
            st.success(f"Ce bien se situe **{diff_pct}% en-dessous** de la médiane de {quartier_sel}.")

# =====================================================================
# ONGLET 2 : RECOMMANDATEUR INTELLIGENT PAR BUDGET
# =====================================================================
with tab2:
    st.subheader("🎯 Trouvez les Quartiers et Biens Accessibles selon votre Budget")
    
    col_b1, col_b2, col_b3 = st.columns([1, 1, 1])
    with col_b1:
        budget_min = st.number_input("Budget Minimum (FCFA)", min_value=30000, max_value=5000000, value=200000, step=50000)
    with col_b2:
        budget_max = st.number_input("Budget Maximum (FCFA)", min_value=50000, max_value=10000000, value=500000, step=50000)
    with col_b3:
        filtre_meuble = st.selectbox("Option Meublé", ["Tous", "Non Meublé Uniquement", "Meublé Uniquement"])

    # Filtrage des annonces de la base Gold
    df_filtered = df_gold[(df_gold['prix_fcfa'] >= budget_min) & (df_gold['prix_fcfa'] <= budget_max)].copy()
    
    if filtre_meuble == "Non Meublé Uniquement":
        df_filtered = df_filtered[df_filtered['is_meuble'] == 0]
    elif filtre_meuble == "Meublé Uniquement":
        df_filtered = df_filtered[df_filtered['is_meuble'] == 1]

    st.markdown(f"#### 📊 Résultat : **{len(df_filtered)} annonces** disponibles dans cette fourchette.")
    
    if not df_filtered.empty:
        col_rec1, col_rec2 = st.columns([1, 1.2])
        
        with col_rec1:
            st.write("##### 📍 Quartiers recommandés (par volume d'offres)")
            q_counts = df_filtered['quartier'].value_counts().head(8).reset_index()
            q_counts.columns = ['Quartier', 'Annonces Disponibles']
            st.dataframe(q_counts, use_container_width=True, hide_index=True)
            
        with col_rec2:
            st.write("##### 💡 Profil de biens accessibles dans ce budget")
            avg_pieces = df_filtered['nb_pieces'].mean()
            pct_piscine = (df_filtered['has_piscine'].mean() * 100)
            pct_parking = (df_filtered['has_parking'].mean() * 100)
            
            st.markdown(f"""
            - **Nombre moyen de pièces :** `{avg_pieces:.1f} pièces`
            - **Taux de présence Parking :** `{pct_parking:.1f}%`
            - **Accès Piscine / Standing :** `{pct_piscine:.1f}%`
            """)
            
        st.divider()
        st.write("##### 📋 Échantillon de biens réels correspondants")
        cols_to_show = ['quartier', 'prix_fcfa', 'nb_pieces', 'nb_salles_bain', 'is_meuble', 'source']
        cols_existing = [c for c in cols_to_show if c in df_filtered.columns]
        
        df_display = df_filtered[cols_existing].head(15).copy()
        df_display['prix_fcfa'] = df_display['prix_fcfa'].apply(lambda x: f"{int(x):,.0f} FCFA".replace(",", " "))
        if 'is_meuble' in df_display.columns:
            df_display['is_meuble'] = df_display['is_meuble'].apply(lambda x: "Oui" if x==1 else "Non")
            
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    else:
        st.warning("Aucun bien ne correspond exactement à ces critères. Essayez d'élargir votre fourchette de budget.")

# =====================================================================
# ONGLET 3 : CARTE INTERACTIVE DU MARCHÉ
# =====================================================================
with tab3:
    st.subheader("🗺️ Distribution Cartographique des Loyers Médians à Dakar")
    
    df_map = df_gold.groupby('quartier').agg(
        prix_median=('prix_fcfa', 'median'),
        nb_annonces=('prix_fcfa', 'count'),
        lat=('latitude', 'mean'),
        lon=('longitude', 'mean')
    ).reset_index()
    
    df_map = df_map[df_map['quartier'] != 'Inconnu'].dropna(subset=['lat', 'lon'])
    
    # Carte Folium
    m = folium.Map(location=[14.720, -17.465], zoom_start=12, tiles="cartodbpositron")
    
    for _, r in df_map.iterrows():
        prix = r['prix_median']
        # Code couleur dynamique selon le standing du quartier
        if prix >= 750000:
            color = "#E74C3C"  # Rouge (Haut standing)
        elif prix >= 400000:
            color = "#F39C12"  # Orange (Moyen-haut standing)
        else:
            color = "#2ECC71"  # Vert (Accessible)
            
        popup_html = f"""
        <div style='font-family: Arial; width: 180px;'>
            <b>{r['quartier']}</b><br>
            • Loyer Médian : <b>{int(prix):,.0f} FCFA</b><br>
            • Volume d'offres : <b>{r['nb_annonces']}</b>
        </div>
        """.replace(",", " ")
        
        folium.CircleMarker(
            location=[r['lat'], r['lon']],
            radius=5 + np.log1p(r['nb_annonces']) * 2.5,
            popup=popup_html,
            color=color,
            fill=True,
            fill_color=color,
            fill_opacity=0.7
        ).add_to(m)
        
    st_folium(m, width=1200, height=520)
    
    st.caption("🔴 **Rouge** : Loyers médians ≥ 750k FCFA | 🟠 **Orange** : 400k à 750k FCFA | 🟢 **Vert** : < 400k FCFA")

# =====================================================================
# ONGLET 4 : GUIDE STRATÉGIQUE DES PLATEFORMES
# =====================================================================
with tab4:
    st.subheader("📢 Analyse des Sources de Données & Guide de Publication")
    
    col_g1, col_g2 = st.columns([1, 1.2])
    
    with col_g1:
        st.write("##### 📊 Répartition des Annonces par Source (Dataset Gold)")
        if 'source' in df_gold.columns:
            source_counts = df_gold['source'].value_counts().reset_index()
            source_counts.columns = ['Source', 'Nombre d\'Annonces']
            
            fig_pie = px.pie(
                source_counts, values='Nombre d\'Annonces', names='Source',
                hole=0.4, color_discrete_sequence=px.colors.qualitative.Set2
            )
            fig_pie.update_layout(margin=dict(t=20, b=20, l=20, r=20))
            st.plotly_chart(fig_pie, use_container_width=True)
            
    with col_g2:
        st.write("##### 💡 Recommandations Stratégiques de Publication")
        st.markdown("""
        **1. Pour les Biens de Haut Standing (> 800 000 FCFA) :**
        - **Plateformes privilégiées :** *Expat-Dakar* & Agences immobilières professionnelles.
        - **Public cible :** Expatriés, cadres du secteur privé, diplomates.
        - **Conseil :** Rédiger des descriptions détaillées (> 150 mots) et mentionner explicitement les équipements de sécurité, groupe électrogène et ascenseur.

        **2. Pour le Marché Intermédiaire & Familial (300 000 à 800 000 FCFA) :**
        - **Plateformes privilégiées :** *CoinAfrique* & *Expat-Dakar*.
        - **Public cible :** Familles locales, professionnels urbains.
        - **Conseil :** Mettre en avant le nombre de salles de bain et la présence d'un parking/garage.

        **3. Pour les Studios & Offres Économiques (< 300 000 FCFA) :**
        - **Plateformes privilégiées :** *Facebook Marketplace* & Groupes *WhatsApp*.
        - **Public cible :** Étudiants, jeunes actifs.
        - **Conseil :** Mettre l'accent sur la proximité des axes de transport (VDN).
        """)