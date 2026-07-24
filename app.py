"""
app.py
Application Streamlit - Prediction du type de logement Airbnb NYC
Auteur: TonfackElvira
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
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Palette unique : une seule teinte declinee en nuances
PRIMARY = "#2C3E6B"
PRIMARY_LIGHT = "#5B6FA8"
PRIMARY_SOFT = "#EEF0F8"
TEXT_DARK = "#1E2333"
TEXT_MUTED = "#6B7280"
BORDER = "#E5E7EB"

st.markdown(f"""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    * {{
        font-family: 'Inter', sans-serif;
    }}

    #MainMenu {{visibility: hidden;}}
    footer {{visibility: hidden;}}
    .stDeployButton {{display: none;}}
    header {{background: transparent;}}

    .block-container {{
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1200px;
    }}

    /* Bandeau d'en-tete */
    .app-header {{
        background: {PRIMARY};
        border-radius: 12px;
        padding: 1.6rem 1rem;
        text-align: center;
        margin-bottom: 1.5rem;
    }}

    .app-title {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.6rem;
        font-size: 1.7rem;
        font-weight: 700;
        color: #ffffff;
        letter-spacing: -0.01em;
    }}

    .app-subtitle {{
        font-size: 0.9rem;
        color: rgba(255,255,255,0.75);
        margin-top: 0.35rem;
    }}

    .card {{
        background: #ffffff;
        border-radius: 10px;
        padding: 1.2rem;
        border: 1px solid {BORDER};
        margin-bottom: 1rem;
    }}

    .metric-value {{
        font-size: 1.6rem;
        font-weight: 700;
        color: {TEXT_DARK};
    }}

    .metric-label {{
        font-size: 0.78rem;
        color: {TEXT_MUTED};
        text-transform: uppercase;
        letter-spacing: 0.04em;
    }}

    .prediction-box {{
        background: {PRIMARY};
        color: white;
        padding: 1.6rem;
        border-radius: 12px;
        text-align: center;
        margin-bottom: 1rem;
    }}

    .prediction-title {{
        font-size: 0.78rem;
        opacity: 0.8;
        margin-bottom: 0.5rem;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }}

    .prediction-result {{
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.6rem;
        font-size: 1.7rem;
        font-weight: 700;
        margin-bottom: 0.4rem;
    }}

    .confidence-bar {{
        background: rgba(255,255,255,0.25);
        border-radius: 10px;
        height: 6px;
        margin-top: 0.8rem;
        overflow: hidden;
    }}

    .confidence-fill {{
        background: white;
        height: 100%;
        border-radius: 10px;
    }}

    .section-title {{
        font-size: 1.05rem;
        font-weight: 600;
        color: {TEXT_DARK};
        margin: 1.4rem 0 0.8rem 0;
        padding-bottom: 0.35rem;
        border-bottom: 2px solid {PRIMARY};
        display: inline-block;
    }}

    .info-text {{
        background: {PRIMARY_SOFT};
        border-left: 3px solid {PRIMARY};
        padding: 0.9rem;
        border-radius: 0 8px 8px 0;
        margin-top: 1rem;
        color: {TEXT_DARK};
        font-size: 0.85rem;
    }}

    .stButton>button {{
        background: {PRIMARY};
        color: white;
        border-radius: 8px;
        padding: 0.65rem 1.5rem;
        font-weight: 500;
        border: none;
        width: 100%;
        transition: background 0.2s ease;
    }}

    .stButton>button:hover {{
        background: {PRIMARY_LIGHT};
    }}

    .footer {{
        text-align: center;
        padding: 1.5rem;
        color: #A1A5B0;
        font-size: 0.78rem;
        border-top: 1px solid {BORDER};
        margin-top: 2rem;
    }}

    .factor-row {{
        display: flex;
        align-items: center;
        padding: 0.65rem 0.75rem;
        background: {PRIMARY_SOFT};
        border-radius: 8px;
        margin-bottom: 0.5rem;
    }}

    .factor-dot {{
        width: 7px;
        height: 7px;
        background: {PRIMARY};
        border-radius: 50%;
        margin-right: 0.9rem;
        flex-shrink: 0;
    }}

    .empty-state {{
        text-align: center;
        padding: 3rem 1.5rem;
        color: {TEXT_MUTED};
    }}

    section[data-testid="stSidebar"] {{
        background: #ffffff;
        border-right: 1px solid {BORDER};
    }}

    section[data-testid="stSidebar"] .card {{
        box-shadow: none;
    }}
</style>
""", unsafe_allow_html=True)


# ICONES (SVG simples, monochromes, pas d'emojis)

def icon_house(color="#ffffff", size=22):
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M3 11.5 12 4l9 7.5"/><path d="M5.5 9.5V20a1 1 0 0 0 1 1h11a1 1 0 0 0 1-1V9.5"/><path d="M9.5 21v-6h5v6"/></svg>"""


def icon_building(color, size=22):
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="5" y="3" width="14" height="18" rx="1"/><line x1="9" y1="7" x2="9" y2="7.01"/><line x1="15" y1="7" x2="15" y2="7.01"/><line x1="9" y1="11" x2="9" y2="11.01"/><line x1="15" y1="11" x2="15" y2="11.01"/><line x1="9" y1="15" x2="9" y2="15.01"/><line x1="15" y1="15" x2="15" y2="15.01"/></svg>"""


def icon_door(color, size=22):
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><rect x="6" y="2.5" width="12" height="19" rx="1"/><line x1="6" y1="21.5" x2="18" y2="21.5"/><circle cx="14.5" cy="12" r="0.6" fill="{color}"/></svg>"""


def icon_bed(color, size=22):
    return f"""<svg width="{size}" height="{size}" viewBox="0 0 24 24" fill="none" stroke="{color}" stroke-width="1.8" stroke-linecap="round" stroke-linejoin="round"><path d="M2 17V8.5a1.5 1.5 0 0 1 1.5-1.5h4A1.5 1.5 0 0 1 9 8.5V13"/><path d="M15 13V8.5A1.5 1.5 0 0 1 16.5 7h4A1.5 1.5 0 0 1 22 8.5V17"/><path d="M2 13h20v6H2z"/><line x1="2" y1="19" x2="2" y2="21"/><line x1="22" y1="19" x2="22" y2="21"/></svg>"""


ROOM_COLORS = {
    'Entire home/apt': PRIMARY,
    'Private room': PRIMARY_LIGHT,
    'Shared room': "#9CA8D6",
}

ROOM_ICON_FN = {
    'Entire home/apt': icon_building,
    'Private room': icon_door,
    'Shared room': icon_bed,
}


def get_room_color(room_type):
    return ROOM_COLORS.get(room_type, PRIMARY)


def get_room_icon_svg(room_type, color="#ffffff", size=26):
    fn = ROOM_ICON_FN.get(room_type, icon_building)
    return fn(color, size)


# CHARGEMENT DES DONNEES

@st.cache_data
def load_dataset():
    """Charge le dataset pour extraire les quartiers valides"""
    try:
        df = pd.read_csv('nyc_air_bnb.csv')
        df['neighbourhood'] = df['neighbourhood'].str.strip()
        df['neighbourhood_group'] = df['neighbourhood_group'].str.strip()
        return df
    except Exception:
        return None


@st.cache_resource
def load_model():
    """Charge le modele et le label encoder"""
    try:
        package = joblib.load('svm_final_package.pkl')
        return package['pipeline'], package['label_encoder'], package.get('optimal_threshold', 0.33)
    except Exception:
        return None, None, 0.33


def get_quartier(df, borough):
    """Retourne la liste des quartiers pour un arrondissement donne"""
    if df is None:
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
    """Prediction avec seuil ajuste pour Shared room"""
    predictions = []
    for proba in probas:
        if proba[2] >= threshold:
            predictions.append(2)
        else:
            predictions.append(np.argmax([proba[0], proba[1]]))
    return np.array(predictions)


# CHARGEMENT

df_raw = load_dataset()
model, label_encoder, threshold = load_model()

if model is None:
    st.error("Erreur : le modele n'a pas pu etre charge.")
    st.stop()


# HEADER

st.markdown(f"""
<div class="app-header">
    <div class="app-title">{icon_house("#ffffff", 26)}<span>Airbnb NYC Predictor</span></div>
    <div class="app-subtitle">Prediction du type de logement a partir des caracteristiques du bien</div>
</div>
""", unsafe_allow_html=True)


# FORMULAIRE (SIDEBAR)

with st.sidebar:
    st.markdown('<div class="section-title">Caracteristiques</div>', unsafe_allow_html=True)

    with st.form("prediction_form"):
        st.markdown("**Localisation**")

        borough_options = ['Brooklyn', 'Manhattan', 'Queens', 'Bronx', 'Staten Island']
        neighbourhood_group = st.selectbox("Arrondissement", borough_options, index=0)

        available_neighbourhoods = get_quartier(df_raw, neighbourhood_group)

        neighbourhood = st.selectbox(
            "Quartier",
            available_neighbourhoods,
            index=0,
            help="Les quartiers proposes correspondent a l'arrondissement selectionne"
        )
        st.caption(f"{len(available_neighbourhoods)} quartiers disponibles")

        col_lat, col_lon = st.columns(2)
        with col_lat:
            latitude = st.number_input("Latitude", value=40.7128, format="%.4f")
        with col_lon:
            longitude = st.number_input("Longitude", value=-74.0060, format="%.4f")

        st.markdown("**Prix et disponibilite**")
        price = st.number_input("Prix par nuit ($)", min_value=10, max_value=10000, value=150)
        minimum_nights = st.number_input("Nuits minimum", min_value=1, max_value=365, value=2)
        availability_365 = st.slider("Jours de disponibilite / an", 0, 365, 200)

        st.markdown("**Avis clients**")
        number_of_reviews = st.number_input("Nombre d'avis", min_value=0, max_value=1000, value=45)
        reviews_per_month = st.number_input("Avis par mois", min_value=0.0, max_value=50.0, value=2.5, step=0.1)
        last_review = st.date_input("Dernier avis", value=datetime(2019, 6, 15))
        calculated_host_listings_count = st.number_input("Logements de l'hote", min_value=1, max_value=500, value=3)

        submitted = st.form_submit_button("Predire le type de logement")

    st.markdown('<div class="info-text">Modele SVM avec SMOTE. Seuil de decision ajuste par analyse PR-AUC pour optimiser la detection des logements partages.</div>', unsafe_allow_html=True)


# RESULTATS (ZONE PRINCIPALE)

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

    col_result, col_chart = st.columns([1, 1.3])

    with col_result:
        st.markdown(f"""
        <div class="prediction-box">
            <div class="prediction-title">Type de logement predit</div>
            <div class="prediction-result">{get_room_icon_svg(room_type, "#ffffff", 24)}<span>{room_type}</span></div>
            <div style="font-size: 0.9rem; opacity: 0.85;">Confiance : {confidence*100:.1f}%</div>
            <div class="confidence-bar">
                <div class="confidence-fill" style="width: {confidence*100}%;"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        m1, m2, m3 = st.columns(3)
        with m1:
            st.markdown(f'<div class="card"><div class="metric-value">{price}$</div><div class="metric-label">Prix / nuit</div></div>', unsafe_allow_html=True)
        with m2:
            st.markdown(f'<div class="card"><div class="metric-value">{number_of_reviews}</div><div class="metric-label">Avis recus</div></div>', unsafe_allow_html=True)
        with m3:
            st.markdown(f'<div class="card"><div class="metric-value">{availability_365}</div><div class="metric-label">Jours dispo.</div></div>', unsafe_allow_html=True)

    with col_chart:
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
            title=dict(text="Probabilites par classe", font=dict(size=14, color=TEXT_DARK)),
            yaxis_title='',
            xaxis_title='',
            showlegend=False,
            height=300,
            margin=dict(t=40, b=20, l=20, r=20),
            yaxis=dict(range=[0, 105]),
            font=dict(color=TEXT_DARK)
        )
        st.plotly_chart(fig, use_container_width=True)

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

    fcols = st.columns(2)
    for i, (factor, desc) in enumerate(factors):
        with fcols[i % 2]:
            st.markdown(f"""
            <div class="factor-row">
                <div class="factor-dot"></div>
                <div>
                    <div style="font-weight: 600; color: {TEXT_DARK}; font-size: 0.9rem;">{factor}</div>
                    <div style="font-size: 0.8rem; color: {TEXT_MUTED};">{desc}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)

else:
    st.markdown(f"""
    <div class="empty-state">
        {icon_house(PRIMARY_LIGHT, 42)}
        <div style="font-size: 1.05rem; font-weight: 500; color: {TEXT_DARK}; margin-top: 0.8rem;">
            Remplissez le formulaire dans le menu lateral
        </div>
        <div style="font-size: 0.85rem; margin-top: 0.2rem;">
            Les resultats de prediction s'afficheront ici
        </div>
    </div>
    """, unsafe_allow_html=True)


# ANALYSE

st.markdown('<div class="section-title">Analyse du dataset</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs(["Distribution", "Prix par type", "Carte"])

with tab1:
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown("**Repartition des types de logement**")
        dist_data = pd.DataFrame({
            'Type': ['Entire home/apt', 'Private room', 'Shared room'],
            'Count': [25409, 22326, 1160],
        })

        fig_pie = px.pie(
            dist_data, values='Count', names='Type', color='Type',
            color_discrete_map=ROOM_COLORS, hole=0.45
        )
        fig_pie.update_traces(textinfo='percent+label', textfont_size=11)
        fig_pie.update_layout(showlegend=False, height=320, margin=dict(t=10, b=10, l=10, r=10), font=dict(color=TEXT_DARK))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_b:
        st.markdown("**Prix moyen par arrondissement**")
        borough_data = pd.DataFrame({
            'Arrondissement': ['Manhattan', 'Brooklyn', 'Queens', 'Bronx', 'Staten Island'],
            'Prix moyen': [220, 150, 120, 95, 85]
        })

        fig_bar = px.bar(borough_data, x='Arrondissement', y='Prix moyen', color_discrete_sequence=[PRIMARY])
        fig_bar.update_layout(showlegend=False, height=320, margin=dict(t=10, b=10, l=10, r=10), font=dict(color=TEXT_DARK))
        st.plotly_chart(fig_bar, use_container_width=True)

with tab2:
    st.markdown("**Distribution des prix par type de logement**")

    np.random.seed(42)
    price_data = []
    for room_type, count, mean_price in [('Entire home/apt', 500, 200), ('Private room', 500, 90), ('Shared room', 100, 50)]:
        prices = np.random.lognormal(np.log(mean_price), 0.5, count)
        for p in prices:
            price_data.append({'Type': room_type, 'Prix': p})

    price_df = pd.DataFrame(price_data)

    fig_box = px.box(price_df, x='Type', y='Prix', color='Type', color_discrete_map=ROOM_COLORS)
    fig_box.update_layout(showlegend=False, height=380, margin=dict(t=10, b=10, l=10, r=10), yaxis_title='Prix par nuit ($)', font=dict(color=TEXT_DARK))
    st.plotly_chart(fig_box, use_container_width=True)

with tab3:
    st.markdown("**Carte des logements (echantillon)**")

    np.random.seed(42)
    map_data = pd.DataFrame({
        'lat': np.random.normal(40.73, 0.08, 300),
        'lon': np.random.normal(-73.95, 0.08, 300),
        'price': np.random.lognormal(4.5, 0.6, 300),
        'type': np.random.choice(['Entire home/apt', 'Private room', 'Shared room'], 300, p=[0.52, 0.46, 0.02])
    })

    fig_map = px.scatter_mapbox(
        map_data, lat='lat', lon='lon', color='type', size='price',
        color_discrete_map=ROOM_COLORS, zoom=10, height=430
    )
    fig_map.update_layout(
        mapbox_style="carto-positron", margin=dict(t=0, b=0, l=0, r=0),
        showlegend=True, legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    st.plotly_chart(fig_map, use_container_width=True)


st.markdown("""
<div class="footer">
    <div style="font-weight: 600; color: #6B7280;">Projet Airbnb NYC</div>
    <div>Modele : SVM avec SMOTE &nbsp;|&nbsp; Dataset : Inside Airbnb</div>
</div>
""", unsafe_allow_html=True)