import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(
    page_title="Analyse Trafic RATP 2021",
    page_icon="🚇",
    layout="wide"
)

# --- 1. CHARGEMENT ET PREPARATION ---
@st.cache_data
def load_data():
    # URL Stable : Trafic annuel entrant par station (2021)
    # C'est un dataset statique très propre, parfait pour éviter les erreurs API
    url = "https://data.ratp.fr/api/explore/v2.1/catalog/datasets/trafic-annuel-entrant-par-station-du-reseau-ferre-2021/exports/csv?lang=fr&timezone=Europe%2FParis&use_labels=true&delimiter=%3B"
   
    try:
        df = pd.read_csv(url, sep=";", on_bad_lines='skip')
       
        # Mapping des colonnes pour simplifier
        # Le dataset contient : Rang, Réseau, Station, Trafic, Correspondance_1... Ville, Arrondissement
        cols_map = {
            'Réseau': 'reseau',
            'Station': 'station',
            'Trafic': 'trafic',
            'Ville': 'ville',
            'Arrondissement pour Paris': 'arrondissement',
            'Correspondance_1': 'c1',
            'Correspondance_2': 'c2',
            'Correspondance_3': 'c3'
        }
        df = df.rename(columns=cols_map)
       
        # Nettoyage
        # On garde principalement le Métro et RER
        if 'reseau' in df.columns:
            df = df[df['reseau'].isin(['Métro', 'RER'])]
           
        # On nettoie la ville (Paris parfois noté différemment)
        df['ville'] = df['ville'].fillna('Inconnu')
       
        # Création de variables "SD3" (Feature Engineering)
        # 1. Complexité : Combien de correspondances ?
        # On compte les colonnes c1, c2, c3... non nulles
        cols_corr = [c for c in df.columns if c.startswith('c') and len(c)==2]
        df['nb_correspondances'] = df[cols_corr].notna().sum(axis=1)
       
        # 2. Catégorie de station (Grosse, Moyenne, Petite) basée sur les quartiles
        q1 = df['trafic'].quantile(0.25)
        q3 = df['trafic'].quantile(0.75)
       
        def categorize(x):
            if x < q1: return 'Petite'
            elif x > q3: return 'Hub Majeur'
            else: return 'Standard'
           
        df['categorie_trafic'] = df['trafic'].apply(categorize)
       
        # 3. Localisation simplifiée
        df['localisation'] = df['ville'].apply(lambda x: 'Paris' if 'Paris' in str(x) else 'Banlieue')

    except Exception as e:
        st.error(f"Erreur API : {e}")
        return pd.DataFrame()

    return df

# --- 2. FONCTIONS GRAPHIQUES ---

def plot_sunburst(df):
    """
    Sunburst Chart : Réseau -> Localisation -> Ville
    Permet de voir la hiérarchie du trafic
    """
    # On regroupe les petites villes pour la lisibilité du graph
    df_sun = df.copy()
    top_villes = df_sun.groupby('ville')['trafic'].sum().nlargest(10).index
    df_sun.loc[~df_sun['ville'].isin(top_villes), 'ville'] = 'Autres Villes'
   
    fig = px.sunburst(
        df_sun,
        path=['reseau', 'localisation', 'ville'],
        values='trafic',
        title="🗺️ Répartition géographique du trafic",
        color='trafic',
        color_continuous_scale='RdBu'
    )
    return fig

def plot_scatter_complexity(df):
    """
    Scatter Plot : Trafic vs Nombre de correspondances
    Question : Les stations avec plus de correspondances ont-elles plus de trafic ?
    """
    fig = px.scatter(
        df,
        x="nb_correspondances",
        y="trafic",
        size="trafic",
        color="reseau",
        hover_name="station",
        log_y=True, # Échelle log car les écarts de trafic sont énormes
        title="🔗 Corrélation : Trafic vs Connectivité (Échelle Log)",
        labels={"nb_correspondances": "Nombre de lignes en correspondance", "trafic": "Trafic annuel"}
    )
    return fig

def plot_top_stations(df):
    """Top 15 classique"""
    df_top = df.sort_values('trafic', ascending=False).head(15)
   
    fig = px.bar(
        df_top,
        x='trafic',
        y='station',
        orientation='h',
        color='reseau',
        title="🏆 Top 15 des Stations les plus fréquentées",
        text_auto='.2s',
        color_discrete_map={'Métro': '#003CA6', 'RER': '#E3051C'}
    )
    fig.update_layout(yaxis={'categoryorder':'total ascending'})
    return fig

def plot_boxplot_reseau(df):
    """Distribution du trafic par réseau"""
    fig = px.box(
        df,
        x="reseau",
        y="trafic",
        color="reseau",
        points="outliers",
        title="📊 Distribution du trafic par Réseau",
        log_y=True
    )
    return fig

# --- 3. MAIN ---

def main():
    st.title("🚇 Dashboard Trafic Annuel RATP")
    st.markdown("Données : **Trafic annuel entrant par station (2021)**. Ce dataset est très fiable.")
   
    df = load_data()
   
    if df.empty:
        st.error("Erreur de chargement.")
        st.stop()

    # --- SIDEBAR ---
    with st.sidebar:
        st.header("Filtres")
        # Filtre Ville
        villes = sorted(df['ville'].unique())
        sel_villes = st.multiselect("Filtrer par Ville", villes)
       
        # Filtre Réseau
        reseaux = sorted(df['reseau'].unique())
        sel_reseau = st.multiselect("Filtrer par Réseau", reseaux, default=reseaux)
       
    # Filtrage
    mask_ville = df['ville'].isin(sel_villes) if sel_villes else True
    mask_reseau = df['reseau'].isin(sel_reseau) if sel_reseau else True
   
    df_filtered = df[mask_ville & mask_reseau]
   
    # --- KPI ---
    c1, c2, c3, c4 = st.columns(4)
   
    total_trafic = df_filtered['trafic'].sum()
    top_station_name = df_filtered.loc[df_filtered['trafic'].idxmax(), 'station']
    avg_trafic = df_filtered['trafic'].mean()
    nb_stations = len(df_filtered)
   
    c1.metric("Trafic Total (2021)", f"{total_trafic:,.0f}".replace(',', ' '))
    c2.metric("Station N°1", top_station_name)
    c3.metric("Moyenne / Station", f"{avg_trafic:,.0f}".replace(',', ' '))
    c4.metric("Nombre de stations", nb_stations)
   
    st.divider()
   
    # --- ONGLETS VIZ ---
    tab1, tab2, tab3 = st.tabs(["Vue Générale", "Analyse Avancée (SD3)", "Données"])
   
    with tab1:
        col_g1, col_g2 = st.columns([1, 1])
        with col_g1:
            st.plotly_chart(plot_top_stations(df_filtered), use_container_width=True)
        with col_g2:
            st.plotly_chart(plot_sunburst(df_filtered), use_container_width=True)
           
    with tab2:
        st.subheader("🔍 Corrélations et Distributions")
        st.markdown("""
        Cette section analyse la relation entre la **connectivité** d'une station et son **trafic**,
        ainsi que la dispersion des données.
        """)
       
        c_adv1, c_adv2 = st.columns(2)
        with c_adv1:
            st.plotly_chart(plot_scatter_complexity(df_filtered), use_container_width=True)
            st.info("Note : L'échelle logarithmique est utilisée car certaines stations (Châtelet, Gare du Nord) écrasent le graphique.")
        with c_adv2:
            st.plotly_chart(plot_boxplot_reseau(df_filtered), use_container_width=True)
           
    with tab3:
        st.dataframe(df_filtered, use_container_width=True)
        csv = df_filtered.to_csv(index=False).encode('utf-8')
        st.download_button("Télécharger les données", csv, "trafic_ratp.csv", "text/csv")

if __name__ == "__main__":
    main()
