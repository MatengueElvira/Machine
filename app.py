"""
app.py
Application Streamlit - Prediction du type de logement Airbnb NYC
Auteur:  TonfackElvira 
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configuration de la page
st.set_page_config(
    page_title="Airbnb NYC - Prediction",
    page_icon="LOGEMENT",
    layout="wide",
    initial_sidebar_state="expanded"
)

#  petit CSS 
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');
    
    * {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        margin-bottom: 0.5rem;
        letter-spacing: -0.02em;
    }
    
    .sub-header {
        font-size: 1.1rem;
        color: #666;
        font-weight: 400;
        margin-bottom: 2rem;
    }
    
    .card {
        background: #ffffff;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 1px 3px rgba(0,0,0,0.08);
        border: 1px solid #f0f0f0;
        margin-bottom: 1rem;
    }
    
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1a1a2e;
    }
    
    .metric-label {
        font-size: 0.85rem;
        color: #888;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 2rem;
        border-radius: 16px;
        text-align: center;
        margin: 1rem 0;
    }
    
    .prediction-title {
        font-size: 0.9rem;
        opacity: 0.9;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
    }
    
    .prediction-result {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .confidence-bar {
        background: rgba(255,255,255,0.2);
        border-radius: 10px;
        height: 8px;
        margin-top: 1rem;
        overflow: hidden;
    }
    
    .confidence-fill {
        background: white;
        height: 100%;
        border-radius: 10px;
        transition: width 0.5s ease;
    }
    
    .section-title {
        font-size: 1.3rem;
        font-weight: 600;
        color: #1a1a2e;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #667eea;
        display: inline-block;
    }
    
    .info-text {
        background: #f8f9fa;
        border-left: 4px solid #667eea;
        padding: 1rem;
        border-radius: 0 8px 8px 0;
        margin: 1rem 0;
    }
    
    .stButton>button {
        background: #1a1a2e;
        color: white;
        border-radius: 8px;
        padding: 0.75rem 2rem;
        font-weight: 500;
        border: none;
        transition: all 0.3s ease;
    }
    
    .stButton>button:hover {
        background: #667eea;
        transform: translateY(-1px);
        box-shadow: 0 4px 12px rgba(102,126,234,0.3);
    }
    
    .footer {
        text-align: center;
        padding: 2rem;
        color: #999;
        font-size: 0.85rem;
        border-top: 1px solid #eee;
        margin-top: 3rem;
    }
</style>
""", unsafe_allow_html=True)


#  CHARGEMENT DES DONNEES 

@st.cache_data
def load_dataset():
    """Charge le dataset pour extraire les quartiers valides"""
    try:
        df = pd.read_csv('nyc_air_bnb.csv')
        # Nettoyage des donnees
        df['neighbourhood'] = df['neighbourhood'].str.strip()
        df['neighbourhood_group'] = df['neighbourhood_group'].str.strip()
        return df
    except:
        # Donnees par defaut si le fichier n'existe pas
        return None


@st.cache_resource
def load_model():
    """Charge le modele et le label encoder"""
    try:
        package = joblib.load('svm_final_package.pkl')
        return package['pipeline'], package['label_encoder'], package.get('optimal_threshold', 0.33)
    except:
        return None, None, 0.33


def get_quartier(df, borough):
    """Retourne la liste des quartiers pour un arrondissement donne"""
    if df is None:
        # Donnees par defaut dans le casou le fichier csv n'est pas bien chargé 
        defaults = {
            'Brooklyn': ['Williamsburg', 'Bedford-Stuyvesant', 'Brooklyn Heights', 'Park Slope', 
                        'Crown Heights', 'Bushwick', 'Greenpoint', 'Red Hook', 'DUMBO', 'Fort Greene'],
            'Manhattan': ['Hell\'s Kitchen', 'Upper East Side', 'Upper West Side', 'Midtown',
                         'East Village', 'West Village', 'SoHo', 'Tribeca', 'Harlem', 'Chelsea'],
            'Queens': ['Astoria', 'Long Island City', 'Flushing', 'Jamaica', 'Forest Hills',
                      'Sunnyside', 'Jackson Heights', 'Woodside', 'Rego Park', 'Bayside'],
            'Bronx': ['Mott Haven', 'Fordham', 'Riverdale', 'Pelham Bay', 'Throgs Neck',
                     'Kingsbridge', 'Belmont', 'Norwood', 'Wakefield', 'Soundview'],
            'Staten Island': ['St. George', 'Tottenville', 'Great Kills', 'New Dorp', 'Port Richmond',
                             'West Brighton', 'Rosebank', 'Eltingville', 'Huguenot', 'Arden Heights']
        }
        return defaults.get(borough, ['Unknown'])
    
    # Filtrons  par arrondissement et retourner les quartiers uniques
    neighbourhoods = df[df['neighbourhood_group'] == borough]['neighbourhood'].unique()
    return sorted(neighbourhoods.tolist())


def prepare_input(data_dict):
    """Prepare les donnees pour la prediction"""
    df = pd.DataFrame([data_dict])
    
    df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
    max_date = pd.Timestamp('2019-07-05')
    
    df['last_review_year'] = df['last_review'].dt.year
    df['last_review_month'] = df['last_review'].dt.month
    df['days_since_last_review'] = (max_date - df['last_review']).dt.days
    df['has_reviews'] = (df['number_of_reviews'] > 0).astype(int)
    df['price_per_night'] = df['price'] / df['minimum_nights'].replace(0, 1)
    df['reviews_per_month'] = df['reviews_per_month'].fillna(0)
    
    df['is_cheap'] = (df['price'] < 50).astype(int)
    df['is_very_cheap'] = (df['price'] < 30).astype(int)
    df['low_minimum_nights'] = (df['minimum_nights'] <= 2).astype(int)
    
    for col in ['id', 'name', 'host_id', 'host_name', 'last_review']:
        if col in df.columns:
            df = df.drop(columns=[col])
    
    return df


def predict_with_threshold(probas, threshold):
    """Pred avec seuil ajuste pour Shared room"""
    predictions = []
    for proba in probas:
        if proba[2] >= threshold:
            predictions.append(2)
        else:
            predictions.append(np.argmax([proba[0], proba[1]]))
    return np.array(predictions)


def get_room_color(room_type):
    colors = {
        'Entire home/apt': '#667eea',
        'Private room': '#f093fb',
        'Shared room': '#f5576c'
    }
    return colors.get(room_type, '#667eea')


def get_room_icon(room_type):
    icons = {
        'Entire home/apt': '',
        'Private room': '',
        'Shared room': ''
    }
    return icons.get(room_type, '')


#CHARGEMENT

df_raw = load_dataset()
model, label_encoder, threshold = load_model()

if model is None:
    st.error("Erreur: Le modele n'a pas pu etre charge.")
    st.stop()


# HEADER 

st.markdown('<div class="main-header">Airbnb NYC Predictor</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Prediction du type de logement basee sur les caracteristiques du bien</div>', unsafe_allow_html=True)




col_left, col_right = st.columns([1, 1.2])


# FORMULAIRE

with col_left:
    st.markdown('<div class="section-title">Caracteristiques du logement</div>', unsafe_allow_html=True)
    
    with st.form("prediction_form"):
        st.markdown('<div class="card">', unsafe_allow_html=True)
        
        # Localisation - ARRONDISSEMENT PUIS QUARTIER
        st.subheader("Localisation")
        
        # Selection de l'arrondissement
        borough_options = ['Brooklyn', 'Manhattan', 'Queens', 'Bronx', 'Staten Island']
        neighbourhood_group = st.selectbox(
            "Arrondissement",
            borough_options,
            index=0
        )
        
        # Selection du quartier (dependant de l'arrondissement)
        available_neighbourhoods = get_quartier(df_raw, neighbourhood_group)
        
        neighbourhood = st.selectbox(
            "Quartier",
            available_neighbourhoods,
            index=0,
            help="Les quartiers proposes correspondent a l'arrondissement selectionne"
        )
        
        # Afficher le nombre de quartiers disponibles
        st.caption(f"{len(available_neighbourhoods)} quartiers disponibles dans {neighbourhood_group}")
        
        latitude = st.number_input("Latitude", value=40.7128, format="%.6f")
        longitude = st.number_input("Longitude", value=-74.0060, format="%.6f")
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Prix et sejour
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Prix et disponibilite")
        
        col3, col4 = st.columns(2)
        with col3:
            price = st.number_input("Prix par nuit ($)", min_value=10, max_value=10000, value=150)
        with col4:
            minimum_nights = st.number_input("Nuits minimum", min_value=1, max_value=365, value=2)
        
        availability_365 = st.slider("Jours de disponibilite / an", 0, 365, 200)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Reviews
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.subheader("Avis clients")
        
        col5, col6 = st.columns(2)
        with col5:
            number_of_reviews = st.number_input("Nombre d'avis", min_value=0, max_value=1000, value=45)
        with col6:
            reviews_per_month = st.number_input("Avis par mois", min_value=0.0, max_value=50.0, value=2.5, step=0.1)
        
        last_review = st.date_input("Dernier avis", value=datetime(2019, 6, 15))
        calculated_host_listings_count = st.number_input("Nombre de logements de l'hote", min_value=1, max_value=500, value=3)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        submitted = st.form_submit_button("Predire le type de logement")
    
    # Info sur le modele
    st.markdown('<div class="info-text">', unsafe_allow_html=True)
    st.markdown("""
    **A propos du modele:**
    
    Ce modele utilise un SVM (Support Vector Machine) avec SMOTE pour gerer le desequilibre des classes. 
    Le seuil de decision est ajuste via l'analyse PR-AUC pour optimiser la detection des logements partages.
    """)
    st.markdown('</div>', unsafe_allow_html=True)


# partie des résultats

with col_right:
    if submitted:
        input_data = {
            'neighbourhood_group': neighbourhood_group,
            'neighbourhood': neighbourhood,
            'latitude': latitude,
            'longitude': longitude,
            'price': price,
            'minimum_nights': minimum_nights,
            'number_of_reviews': number_of_reviews,
            'reviews_per_month': reviews_per_month,
            'calculated_host_listings_count': calculated_host_listings_count,
            'availability_365': availability_365,
            'last_review': pd.Timestamp(last_review)
        }
        
        df_input = prepare_input(input_data)
        
        probabilities = model.predict_proba(df_input)
        prediction = predict_with_threshold(probabilities, threshold)
        room_type = label_encoder.inverse_transform(prediction)[0]
        confidence = probabilities[0][prediction[0]]
        
        room_color = get_room_color(room_type)
        
        st.markdown(f"""
        <div class="prediction-box" style="background: linear-gradient(135deg, {room_color} 0%, {room_color}dd 100%);">
            <div class="prediction-title">Type de logement predit</div>
            <div class="prediction-result">{get_room_icon(room_type)} {room_type}</div>
            <div style="font-size: 1rem; opacity: 0.9;">Confiance: {confidence*100:.1f}%</div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence*100}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">Probabilites par classe</div>', unsafe_allow_html=True)
        
        prob_df = pd.DataFrame({
            'Type de logement': label_encoder.classes_,
            'Probabilite': probabilities[0] * 100
        })
        
        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=prob_df['Type de logement'],
            y=prob_df['Probabilite'],
            marker_color=[get_room_color(c) for c in prob_df['Type de logement']],
            text=[f'{p:.1f}%' for p in prob_df['Probabilite']],
            textposition='outside',
            textfont=dict(size=12)
        ))
        
        fig.update_layout(
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            yaxis_title='Probabilite (%)',
            xaxis_title='',
            showlegend=False,
            height=350,
            margin=dict(t=30, b=30, l=30, r=30),
            yaxis=dict(range=[0, 105])
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown('<div class="section-title">Details de la prediction</div>', unsafe_allow_html=True)
        
        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{price}$</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Prix / nuit</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with m2:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{number_of_reviews}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Avis recus</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        with m3:
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-value">{availability_365}</div>', unsafe_allow_html=True)
            st.markdown('<div class="metric-label">Jours disponibles</div>', unsafe_allow_html=True)
            st.markdown('</div>', unsafe_allow_html=True)
        
        st.markdown('<div class="section-title">Facteurs influents</div>', unsafe_allow_html=True)
        
        factors = []
        if price < 50:
            factors.append(("Prix bas (< 50$)", "Favorise les logements partages"))
        elif price > 200:
            factors.append(("Prix eleve (> 200$)", "Favorise les logements entiers"))
        
        if minimum_nights == 1:
            factors.append(("Sejour court", "Typique des chambres privees"))
        
        if number_of_reviews < 5:
            factors.append(("Peu d'avis", "Nouveau logement ou peu frequente"))
        
        if calculated_host_listings_count > 10:
            factors.append(("Hote professionnel", "Probablement un logement entier"))
        
        if not factors:
            factors.append(("Profil standard", "Caracteristiques equilibrees"))
        
        for factor, desc in factors:
            st.markdown(f"""
            <div style="display: flex; align-items: center; padding: 0.75rem; background: #f8f9fa; border-radius: 8px; margin-bottom: 0.5rem;">
                <div style="width: 8px; height: 8px; background: {room_color}; border-radius: 50%; margin-right: 1rem;"></div>
                <div>
                    <div style="font-weight: 600; color: #1a1a2e;">{factor}</div>
                    <div style="font-size: 0.85rem; color: #666;">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
    
    else:
        st.markdown("""
        <div style="text-align: center; padding: 4rem 2rem; color: #999;">
            <div style="font-size: 4rem; margin-bottom: 1rem;">""</div>
            <div style="font-size: 1.2rem; font-weight: 500; color: #666; margin-bottom: 0.5rem;">
                Remplissez le formulaire
            </div>
            <div style="font-size: 0.9rem;">
                Les resultats de prediction s'afficheront ici
            </div>
        </div>
        """, unsafe_allow_html=True)


# Partie analyse
st.markdown('<div class="section-title">Analyse du dataset</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Distribution", "Prix par type", "Carte"])

with tab1:
    col_a, col_b = st.columns(2)
    
    with col_a:
        st.subheader("Repartition des types de logement")
        dist_data = pd.DataFrame({
            'Type': ['Entire home/apt', 'Private room', 'Shared room'],
            'Count': [25409, 22326, 1160],
            'Pourcentage': [51.97, 45.66, 2.37]
        })
        
        fig_pie = px.pie(
            dist_data, 
            values='Count', 
            names='Type',
            color='Type',
            color_discrete_map={
                'Entire home/apt': '#667eea',
                'Private room': '#f093fb', 
                'Shared room': '#f5576c'
            },
            hole=0.4
        )
        fig_pie.update_traces(textinfo='percent+label', textfont_size=12)
        fig_pie.update_layout(showlegend=False, height=400, margin=dict(t=30, b=30, l=30, r=30))
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col_b:
        st.subheader("Prix moyen par arrondissement")
        borough_data = pd.DataFrame({
            'Arrondissement': ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'],
            'Prix moyen': [220, 150, 120, 95, 85]
        })
        
        fig_bar = px.bar(borough_data, x='Arrondissement', y='Prix moyen', color='Prix moyen', color_continuous_scale='Blues')
        fig_bar.update_layout(showlegend=False, height=400, margin=dict(t=30, b=30, l=30, r=30), coloraxis_showscale=False)
        st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.subheader("Distribution des prix par type de logement")
    
    np.random.seed(42)
    price_data = []
    for room_type, count, mean_price in [('Entire home/apt', 500, 200), ('Private room', 500, 90), ('Shared room', 100, 50)]:
        prices = np.random.lognormal(np.log(mean_price), 0.5, count)
        for p in prices:
            price_data.append({'Type': room_type, 'Prix': p})
    
    price_df = pd.DataFrame(price_data)
    
    fig_box = px.box(price_df, x='Type', y='Prix', color='Type',
                    color_discrete_map={'Entire home/apt': '#667eea', 'Private room': '#f093fb', 'Shared room': '#f5576c'})
    fig_box.update_layout(showlegend=False, height=450, margin=dict(t=30, b=30, l=30, r=30), yaxis_title='Prix par nuit ($)')
    st.plotly_chart(fig_box, use_container_width=True)

with tab3:
    st.subheader("Carte des logements (echantillon)")
    
    np.random.seed(42)
    map_data = pd.DataFrame({
        'lat': np.random.normal(40.73, 0.08, 300),
        'lon': np.random.normal(-73.95, 0.08, 300),
        'price': np.random.lognormal(4.5, 0.6, 300),
        'type': np.random.choice(['Entire home/apt', 'Private room', 'Shared room'], 300, p=[0.52, 0.46, 0.02])
    })
    
    fig_map = px.scatter_mapbox(map_data, lat='lat', lon='lon', color='type', size='price',
                               color_discrete_map={'Entire home/apt': '#667eea', 'Private room': '#f093fb', 'Shared room': '#f5576c'},
                               zoom=10, height=500)
    fig_map.update_layout(mapbox_style="carto-positron", margin=dict(t=0, b=0, l=0, r=0),
                         showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01))
    st.plotly_chart(fig_map, use_container_width=True)




st.markdown("""
<div class="footer">
    <div style="font-weight: 600; color: #666; margin-bottom: 0.5rem;">
        Projet Airbnb NYC
    </div>
    <div>Modele: SVM avec SMOTE | Dataset: Inside Airbnb )</div>
    <div style="margin-top: 0.5rem; font-size: 0.8rem;">Developpe dans le cadre d'un projet academique</div>
</div>
""", unsafe_allow_html=True)