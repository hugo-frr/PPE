"""
Tableau de bord interactif PPE — Soutenance.
Lancer depuis le dossier ppe/ :
    streamlit run src/demo.py
"""
from __future__ import annotations

import sys
from pathlib import Path

# ── résolution du chemin src/ ────────────────────────────────────────────────
_SRC_DIR = Path(__file__).resolve().parent
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from config import MODELS_DIR, PROCESSED_DIR, REPORTS_DIR

# ── configuration de la page ─────────────────────────────────────────────────
st.set_page_config(
    page_title="PPE — Prédiction Airbus",
    page_icon="✈️",
    layout="wide",
)

# ── fonctions utilitaires ─────────────────────────────────────────────────────

def _artifacts_ready() -> bool:
    return (
        (REPORTS_DIR / "predictions_merged.csv").exists()
        and (REPORTS_DIR / "model_comparison.csv").exists()
    )


@st.cache_data(show_spinner=False)
def _load_predictions() -> pd.DataFrame:
    df = pd.read_csv(REPORTS_DIR / "predictions_merged.csv", parse_dates=["Date"])
    return df.sort_values("Date").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def _load_metrics() -> pd.DataFrame:
    return pd.read_csv(REPORTS_DIR / "model_comparison.csv")


@st.cache_data(show_spinner=False)
def _load_market_data() -> pd.DataFrame | None:
    path = PROCESSED_DIR / "market_data_processed.csv"
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    return df.set_index("Date").sort_index()


def _run_pipeline(skip_download: bool, force_retrain: bool) -> None:
    from data.process_data import process_and_save
    from evaluation.metrics import evaluate_models

    with st.status("Exécution du pipeline…", expanded=True) as status:
        if not skip_download:
            from data.download_data import download_market_data
            st.write("📡 Téléchargement Yahoo Finance…")
            download_market_data()
        st.write("🔧 Prétraitement des données…")
        process_and_save()
        st.write("🧠 Entraînement & évaluation des modèles LSTM…")
        evaluate_models(force_retrain=force_retrain)
        status.update(label="✅ Pipeline terminé !", state="complete")

    st.cache_data.clear()
    st.rerun()


# ── barre latérale ────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Contrôles")

    skip_dl = st.checkbox(
        "Ignorer le téléchargement",
        value=_artifacts_ready(),
        help="Utiliser les données brutes déjà téléchargées.",
    )
    force_rt = st.toggle(
        "Forcer le réentraînement",
        value=False,
        help="Réentraîne même si des modèles sauvegardés existent.",
    )

    if st.button("▶ Lancer le pipeline", type="primary", use_container_width=True):
        _run_pipeline(skip_download=skip_dl, force_retrain=force_rt)

    st.divider()

    if _artifacts_ready():
        metrics_df_side = _load_metrics()
        b = metrics_df_side[metrics_df_side["model"] == "baseline_lstm"].iloc[0]
        e = metrics_df_side[metrics_df_side["model"] == "enriched_lstm"].iloc[0]
        st.caption("**Résultats actuels**")
        st.caption(f"Baseline  — RMSE {b['rmse']:.2f} €")
        st.caption(f"Enrichi   — RMSE {e['rmse']:.2f} €")

    st.divider()
    st.caption("PPE · LSTM Baseline vs Enrichi")
    st.caption("Prédiction du cours Airbus (AIR.PA)")


# ── en-tête principal ─────────────────────────────────────────────────────────
st.title("✈️ PPE — Prédiction du Cours Airbus")
st.caption(
    "Comparaison d'un LSTM univarié (baseline) et d'un LSTM multivarié enrichi "
    "avec données corrélées (LVMH, Stellantis, CAC40) et événements tarifaires."
)

if not _artifacts_ready():
    st.info(
        "Aucun résultat disponible. Cliquez sur **▶ Lancer le pipeline** dans la barre latérale.",
        icon="ℹ️",
    )
    st.stop()

# ── chargement ────────────────────────────────────────────────────────────────
pred_df = _load_predictions()
metrics_df = _load_metrics()
market_df = _load_market_data()

b_row = metrics_df[metrics_df["model"] == "baseline_lstm"].iloc[0]
e_row = metrics_df[metrics_df["model"] == "enriched_lstm"].iloc[0]

# ── cartes KPI ────────────────────────────────────────────────────────────────
c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("RMSE Baseline", f"{b_row['rmse']:.2f} €")
c2.metric(
    "RMSE Enrichi",
    f"{e_row['rmse']:.2f} €",
    delta=f"{e_row['rmse'] - b_row['rmse']:+.2f}",
    delta_color="inverse",
)
c3.metric("MAE Baseline", f"{b_row['mae']:.2f} €")
c4.metric(
    "MAE Enrichi",
    f"{e_row['mae']:.2f} €",
    delta=f"{e_row['mae'] - b_row['mae']:+.2f}",
    delta_color="inverse",
)
c5.metric("MAPE Baseline", f"{b_row['mape'] * 100:.2f} %")
c6.metric(
    "MAPE Enrichi",
    f"{e_row['mape'] * 100:.2f} %",
    delta=f"{(e_row['mape'] - b_row['mape']) * 100:+.2f} %",
    delta_color="inverse",
)

st.divider()

# ── onglets ───────────────────────────────────────────────────────────────────
tab_pred, tab_zoom, tab_metrics, tab_data = st.tabs(
    ["📈 Prédictions", "🔍 Zoom & Erreurs", "📊 Métriques", "📋 Données de marché"]
)

# ── onglet 1 : courbe complète ────────────────────────────────────────────────
with tab_pred:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=pred_df["Date"], y=pred_df["y_true"],
        name="Cours réel",
        line=dict(color="#1f77b4", width=2),
    ))
    fig.add_trace(go.Scatter(
        x=pred_df["Date"], y=pred_df["y_pred_baseline"],
        name="LSTM Baseline",
        line=dict(color="#ff7f0e", width=1.5, dash="dot"),
    ))
    fig.add_trace(go.Scatter(
        x=pred_df["Date"], y=pred_df["y_pred_enriched"],
        name="LSTM Enrichi",
        line=dict(color="#2ca02c", width=1.5, dash="dash"),
    ))
    fig.update_layout(
        title="Prédictions sur l'ensemble de test (20 % des données)",
        xaxis_title="Date",
        yaxis_title="Prix Airbus (€)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=500,
    )
    st.plotly_chart(fig, use_container_width=True)

    with st.expander("ℹ️ Protocole d'évaluation"):
        st.markdown(
            "- **Split temporel strict** 80 % train / 20 % test (pas de mélange aléatoire)\n"
            "- **Normalisation** MinMaxScaler ajusté uniquement sur le train\n"
            "- **Séquences** de 60 jours glissants en entrée\n"
            "- **EarlyStopping** patience=4 sur la val_loss (10 % du train)"
        )

# ── onglet 2 : zoom + erreurs ─────────────────────────────────────────────────
with tab_zoom:
    n_days = st.slider(
        "Nombre de jours à afficher",
        min_value=30,
        max_value=len(pred_df),
        value=min(90, len(pred_df)),
        step=10,
    )
    zoom_df = pred_df.tail(n_days).copy()

    fig_z = go.Figure()
    fig_z.add_trace(go.Scatter(
        x=zoom_df["Date"], y=zoom_df["y_true"],
        name="Cours réel", line=dict(color="#1f77b4", width=2),
    ))
    fig_z.add_trace(go.Scatter(
        x=zoom_df["Date"], y=zoom_df["y_pred_baseline"],
        name="LSTM Baseline", line=dict(color="#ff7f0e", width=1.5, dash="dot"),
    ))
    fig_z.add_trace(go.Scatter(
        x=zoom_df["Date"], y=zoom_df["y_pred_enriched"],
        name="LSTM Enrichi", line=dict(color="#2ca02c", width=1.5, dash="dash"),
    ))
    fig_z.update_layout(
        title=f"Zoom — {n_days} derniers jours de test",
        xaxis_title="Date", yaxis_title="Prix Airbus (€)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400,
    )
    st.plotly_chart(fig_z, use_container_width=True)

    # Analyse des erreurs
    zoom_df["err_baseline"] = zoom_df["y_pred_baseline"] - zoom_df["y_true"]
    zoom_df["err_enriched"] = zoom_df["y_pred_enriched"] - zoom_df["y_true"]

    fig_err = go.Figure()
    fig_err.add_trace(go.Bar(
        x=zoom_df["Date"], y=zoom_df["err_baseline"],
        name="Erreur Baseline",
        marker_color="#ff7f0e", opacity=0.7,
    ))
    fig_err.add_trace(go.Bar(
        x=zoom_df["Date"], y=zoom_df["err_enriched"],
        name="Erreur Enrichi",
        marker_color="#2ca02c", opacity=0.7,
    ))
    fig_err.add_hline(y=0, line_color="black", line_width=1)
    fig_err.update_layout(
        title="Erreur de prédiction (prédiction − cours réel)",
        xaxis_title="Date", yaxis_title="Erreur (€)",
        barmode="group", height=300,
    )
    st.plotly_chart(fig_err, use_container_width=True)

# ── onglet 3 : métriques ──────────────────────────────────────────────────────
with tab_metrics:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Tableau comparatif")
        disp = pd.DataFrame({
            "Modèle":    ["LSTM Baseline", "LSTM Enrichi"],
            "RMSE (€)":  [f"{b_row['rmse']:.4f}", f"{e_row['rmse']:.4f}"],
            "MAE (€)":   [f"{b_row['mae']:.4f}", f"{e_row['mae']:.4f}"],
            "MAPE":      [f"{b_row['mape']*100:.2f} %", f"{e_row['mape']*100:.2f} %"],
        })
        st.dataframe(disp, use_container_width=True, hide_index=True)
        st.markdown(
            "**RMSE** — pénalise les grandes erreurs  \n"
            "**MAE** — erreur absolue moyenne en €  \n"
            "**MAPE** — erreur relative en % du cours réel"
        )

    with col_right:
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="RMSE",
            x=["LSTM Baseline", "LSTM Enrichi"],
            y=[b_row["rmse"], e_row["rmse"]],
            marker_color=["#ff7f0e", "#2ca02c"],
        ))
        fig_bar.add_trace(go.Bar(
            name="MAE",
            x=["LSTM Baseline", "LSTM Enrichi"],
            y=[b_row["mae"], e_row["mae"]],
            marker_color=["#ff7f0e", "#2ca02c"],
            opacity=0.55,
        ))
        fig_bar.update_layout(
            title="RMSE et MAE par modèle",
            yaxis_title="Erreur (€)",
            barmode="group",
            height=350,
        )
        st.plotly_chart(fig_bar, use_container_width=True)

    # Scatter réel vs prédit
    st.subheader("Réel vs Prédit — diagramme de dispersion")
    min_v = pred_df["y_true"].min()
    max_v = pred_df["y_true"].max()
    sc_left, sc_right = st.columns(2)

    for col_sc, col_pred, title, color in [
        (sc_left,  "y_pred_baseline", "LSTM Baseline", "#ff7f0e"),
        (sc_right, "y_pred_enriched", "LSTM Enrichi",  "#2ca02c"),
    ]:
        fig_sc = go.Figure()
        fig_sc.add_trace(go.Scatter(
            x=pred_df["y_true"], y=pred_df[col_pred],
            mode="markers",
            marker=dict(color=color, opacity=0.35, size=4),
            name=title,
        ))
        fig_sc.add_trace(go.Scatter(
            x=[min_v, max_v], y=[min_v, max_v],
            mode="lines",
            line=dict(color="black", dash="dash"),
            name="Prédiction parfaite",
        ))
        fig_sc.update_layout(
            title=title,
            xaxis_title="Cours réel (€)",
            yaxis_title="Cours prédit (€)",
            height=380,
        )
        col_sc.plotly_chart(fig_sc, use_container_width=True)

# ── onglet 4 : données de marché ──────────────────────────────────────────────
with tab_data:
    if market_df is None:
        st.warning("Données de marché non disponibles. Lancez le pipeline d'abord.")
        st.stop()

    price_cols = [c for c in market_df.columns if c.endswith("_price")]

    # Performances relatives base 100
    norm_df = market_df[price_cols].dropna()
    norm_df = norm_df / norm_df.iloc[0] * 100

    fig_mkt = go.Figure()
    _COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
    _LABELS = {c: c.replace("_price", "") for c in price_cols}
    for i, col in enumerate(price_cols):
        fig_mkt.add_trace(go.Scatter(
            x=norm_df.index, y=norm_df[col],
            name=_LABELS[col],
            line=dict(color=_COLORS[i % len(_COLORS)]),
        ))
    fig_mkt.update_layout(
        title="Performance relative (base 100 au 01/01/2018)",
        xaxis_title="Date",
        yaxis_title="Indice (base 100)",
        hovermode="x unified",
        height=450,
    )
    st.plotly_chart(fig_mkt, use_container_width=True)

    col_s1, col_s2 = st.columns(2)
    with col_s1:
        st.subheader("Statistiques descriptives — Prix")
        st.dataframe(
            market_df[price_cols].describe().round(2),
            use_container_width=True,
        )
    with col_s2:
        st.subheader("Corrélations — Log-rendements")
        ret_cols = [c for c in market_df.columns if c.endswith("_ret")]
        corr = market_df[ret_cols].corr().round(3)
        st.dataframe(corr, use_container_width=True)
