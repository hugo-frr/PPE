"""
arima_model.py
==============
Pipeline ARIMA / ARIMAX pour la prévision de rendements d'actions françaises.

Améliorations vs version initiale :
- Walk-forward optimisé via statsmodels apply() — O(n) au lieu de O(n²)
- Support ARIMAX (variables exogènes : tariff_dummy, tariff_intensity, macro)
- MAPE supprimé sur les rendements (non défini quand y=0)
- Directional Accuracy ajoutée (métrique clé pour un PPE quant)
- Event-window accuracy : performance spécifique autour des chocs tarifaires
- Résultats exportés en CSV pour la comparaison inter-modèles
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

warnings.filterwarnings("ignore")

try:
    from pmdarima import auto_arima
    PMDARIMA_AVAILABLE = True
except ImportError:
    PMDARIMA_AVAILABLE = False


# ──────────────────────────────────────────────────────────────────────────────
# I. CHARGEMENT ET PRÉPARATION DES DONNÉES
# ──────────────────────────────────────────────────────────────────────────────

def load_and_prepare(
    filepath: str,
    date_col: str = "date",
    close_col: str = "close",
    exog_cols: Optional[List[str]] = None,
) -> Tuple[pd.Series, Optional[pd.DataFrame]]:
    """
    Charge un CSV, construit les log-rendements et extrait les variables exogènes.

    Retourne
    --------
    log_returns : pd.Series
        Log-rendements journaliers (stationnarité vérifiée par ADF).
    exog : pd.DataFrame ou None
        Variables exogènes alignées sur log_returns (si exog_cols fournis).
    """
    df = pd.read_csv(filepath)

    # Normalisation des noms de colonnes (minuscules)
    df.columns = [c.lower().strip() for c in df.columns]
    date_col = date_col.lower()
    close_col = close_col.lower()

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col).set_index(date_col)
    df.index.name = "date"

    # Prix de clôture → log-rendements
    prices = df[close_col].astype(float)
    prices = prices.asfreq("B", method="ffill")          # fréquence régulière jours ouvrés
    log_returns = np.log(prices).diff().dropna()

    # Variables exogènes (tariff features, macro)
    exog = None
    if exog_cols:
        available = [c for c in exog_cols if c in df.columns]
        if available:
            exog = df[available].asfreq("B", method="ffill")
            exog = exog.loc[log_returns.index]            # alignement strict
            exog = exog.fillna(0)

    return log_returns, exog


def load_tariff_events(filepath: str) -> pd.DataFrame:
    """
    Charge le fichier CSV des événements tarifaires.

    Format attendu du CSV :
        date, event_label, tariff_rate, affected_sectors, notes
        2018-03-01, Steel/Aluminum 25%, 0.25, Industrials, ...

    Retourne un DataFrame indexé par date.
    """
    events = pd.read_csv(filepath)
    events.columns = [c.lower().strip() for c in events.columns]
    events["date"] = pd.to_datetime(events["date"])
    events = events.sort_values("date").set_index("date")
    return events


def build_tariff_features(
    index: pd.DatetimeIndex,
    events: pd.DataFrame,
    lags: List[int] = [1, 5, 10, 30],
    window: int = 30,
) -> pd.DataFrame:
    """
    Construit les features tarifaires à partir de l'événementiel.

    Features produites
    ------------------
    tariff_dummy        : 1 le jour de l'annonce, 0 sinon
    tariff_intensity    : valeur du tarif annoncé (ex. 0.25 pour 25 %)
    tariff_lag_{n}      : dummy décalée de n jours ouvrés
    tariff_rolling_{w}  : nb d'annonces sur les w derniers jours
    """
    feat = pd.DataFrame(index=index)
    feat["tariff_dummy"] = 0.0
    feat["tariff_intensity"] = 0.0

    for date, row in events.iterrows():
        if date in feat.index:
            feat.loc[date, "tariff_dummy"] = 1.0
            feat.loc[date, "tariff_intensity"] = float(row.get("tariff_rate", 0))

    for lag in lags:
        feat[f"tariff_lag_{lag}"] = feat["tariff_dummy"].shift(lag, fill_value=0)

    feat[f"tariff_rolling_{window}"] = (
        feat["tariff_dummy"].rolling(window, min_periods=1).sum()
    )

    return feat.fillna(0)


# ──────────────────────────────────────────────────────────────────────────────
# II. STATIONNARITÉ
# ──────────────────────────────────────────────────────────────────────────────

def adf_test(series: pd.Series, label: str = "") -> Dict[str, float]:
    """Test ADF augmenté. Affiche le résultat et retourne les statistiques."""
    result = adfuller(series.dropna(), autolag="AIC")
    stats = {
        "adf_stat":  result[0],
        "p_value":   result[1],
        "crit_1%":   result[4]["1%"],
        "crit_5%":   result[4]["5%"],
        "crit_10%":  result[4]["10%"],
    }
    verdict = "STATIONNAIRE ✓" if result[1] < 0.05 else "NON STATIONNAIRE ✗"
    prefix = f"[{label}] " if label else ""
    print(f"{prefix}ADF test → p={result[1]:.4f} | {verdict}")
    return stats


# ──────────────────────────────────────────────────────────────────────────────
# III. SÉLECTION DE L'ORDRE ARIMA
# ──────────────────────────────────────────────────────────────────────────────

def select_order(
    train: pd.Series,
    exog_train: Optional[pd.DataFrame] = None,
    max_p: int = 4,
    max_q: int = 4,
) -> Tuple[int, int, int]:
    """
    Sélectionne (p, d, q) par auto_arima (pmdarima) ou grid-search AIC.

    Note : sur des log-rendements, d=0 est généralement optimal (déjà stationnaires).
    """
    if PMDARIMA_AVAILABLE:
        m = auto_arima(
            train,
            exogenous=exog_train,
            start_p=0, start_q=0,
            max_p=max_p, max_q=max_q,
            d=0,                       # rendements déjà stationnaires
            seasonal=False,
            information_criterion="aic",
            stepwise=True,
            error_action="ignore",
            suppress_warnings=True,
            trace=False,
        )
        print(f"  auto_arima → ordre optimal : {m.order}")
        return m.order

    # Fallback : grid search AIC
    best_aic, best_order = np.inf, (1, 0, 1)
    for p in range(max_p + 1):
        for q in range(max_q + 1):
            try:
                fit = ARIMA(train, exog=exog_train, order=(p, 0, q)).fit()
                if fit.aic < best_aic:
                    best_aic, best_order = fit.aic, (p, 0, q)
            except Exception:
                continue
    print(f"  grid-search AIC → ordre optimal : {best_order}")
    return best_order


# ──────────────────────────────────────────────────────────────────────────────
# IV. WALK-FORWARD OPTIMISÉ
# ──────────────────────────────────────────────────────────────────────────────

def walk_forward_optimized(
    full_returns: pd.Series,
    test_index: pd.DatetimeIndex,
    order: Tuple[int, int, int],
    full_exog: Optional[pd.DataFrame] = None,
) -> pd.Series:
    """
    Walk-forward J+1 avec expanding window OPTIMISÉ.

    Stratégie : on fit une seule fois sur le train initial, puis on utilise
    model.apply() pour mettre à jour l'état sans re-fitter entièrement.
    Réduit la complexité de O(n²) à O(n).

    Paramètres
    ----------
    full_returns : pd.Series
        Série complète (train + test).
    test_index : pd.DatetimeIndex
        Dates de la période de test.
    order : tuple
        Ordre ARIMA (p, d, q).
    full_exog : pd.DataFrame, optional
        Variables exogènes sur toute la période (train + test).
    """
    train_end = test_index[0]
    train_returns = full_returns.loc[:train_end].iloc[:-1]

    exog_train = full_exog.loc[train_returns.index] if full_exog is not None else None
    exog_test  = full_exog.loc[test_index] if full_exog is not None else None

    preds = []

    try:
        # Fit initial sur le train complet
        model = ARIMA(train_returns, exog=exog_train, order=order)
        fitted = model.fit()

        for i, t in enumerate(test_index):
            # Prévision J+1
            exog_step = exog_test.iloc[[i]] if exog_test is not None else None
            try:
                fc = fitted.forecast(steps=1, exog=exog_step)
                preds.append(float(fc.iloc[0]))
            except Exception:
                preds.append(0.0)

            # Mise à jour incrémentale (append observation réelle)
            new_obs = full_returns.loc[t]
            new_exog = exog_test.iloc[[i]] if exog_test is not None else None
            try:
                fitted = fitted.append(endog=[new_obs], exog=new_exog, refit=False)
            except Exception:
                # Si append échoue, on re-fit sur historique étendu (rare)
                hist = full_returns.loc[:t]
                exog_hist = full_exog.loc[hist.index] if full_exog is not None else None
                fitted = ARIMA(hist, exog=exog_hist, order=order).fit()

    except Exception as e:
        print(f"  Walk-forward échoué ({e}), fallback sur prévisions nulles.")
        preds = [0.0] * len(test_index)

    return pd.Series(preds, index=test_index, name="arima_pred_returns")


# ──────────────────────────────────────────────────────────────────────────────
# V. RECONSTRUCTION DES PRIX
# ──────────────────────────────────────────────────────────────────────────────

def reconstruct_prices(
    anchor_price: float,
    predicted_returns: pd.Series,
) -> pd.Series:
    """Reconstruit les prix à partir des log-rendements : P_t = P_{t-1} * exp(r_t)."""
    prices = [anchor_price * np.exp(predicted_returns.iloc[0])]
    for r in predicted_returns.iloc[1:]:
        prices.append(prices[-1] * np.exp(r))
    return pd.Series(prices, index=predicted_returns.index, name="arima_pred_prices")


# ──────────────────────────────────────────────────────────────────────────────
# VI. MÉTRIQUES
# ──────────────────────────────────────────────────────────────────────────────

def compute_metrics(
    y_true: pd.Series,
    y_pred: pd.Series,
    label: str = "",
    events_index: Optional[pd.DatetimeIndex] = None,
    window_days: int = 5,
) -> Dict[str, float]:
    """
    Calcule les métriques de performance.

    Métriques standards
    -------------------
    RMSE, MAE : sur les log-rendements prédits

    Métriques spécifiques à ce PPE
    --------------------------------
    directional_accuracy : % de jours où le signe de r_pred == signe de r_real
    event_window_accuracy : directional accuracy sur ±{window_days} j autour des chocs
    """
    y_true, y_pred = y_true.dropna(), y_pred.reindex(y_true.index).dropna()
    common = y_true.index.intersection(y_pred.index)
    y_true, y_pred = y_true.loc[common], y_pred.loc[common]

    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))

    # Directional Accuracy
    correct_dir = (np.sign(y_true) == np.sign(y_pred)).mean()
    da = float(correct_dir)

    metrics = {"RMSE": rmse, "MAE": mae, "Directional_Accuracy": da}

    # Event-window Accuracy (métrique académique clé)
    if events_index is not None:
        event_dates = []
        for ed in events_index:
            # ±window_days jours ouvrés autour de chaque choc
            window = pd.bdate_range(
                start=ed - pd.Timedelta(days=window_days * 2),
                end=ed + pd.Timedelta(days=window_days * 2),
            )
            event_dates.extend(window.tolist())
        event_mask = y_true.index.isin(event_dates)
        if event_mask.sum() > 0:
            ewa = float((np.sign(y_true[event_mask]) == np.sign(y_pred[event_mask])).mean())
            metrics["Event_Window_Accuracy"] = ewa

    if label:
        print(f"\n── Métriques ARIMA [{label}] ──")
        for k, v in metrics.items():
            print(f"   {k}: {v:.6f}")

    return metrics


# ──────────────────────────────────────────────────────────────────────────────
# VII. VISUALISATION
# ──────────────────────────────────────────────────────────────────────────────

def plot_predictions(
    real_prices: pd.Series,
    pred_prices: pd.Series,
    events: Optional[pd.DataFrame] = None,
    title: str = "ARIMA — Prix réels vs prédits",
    save_path: Optional[str] = None,
) -> None:
    """
    Graphique prix réels vs prédits avec overlay des chocs tarifaires.
    Les dates d'annonces sont marquées par des lignes verticales rouges.
    """
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                                    gridspec_kw={"height_ratios": [3, 1]})

    # Panneau supérieur : prix
    ax1.plot(real_prices.index, real_prices.values,
             label="Prix réels", color="#1f77b4", linewidth=1.8)
    ax1.plot(pred_prices.index, pred_prices.values,
             label="ARIMA prédit", color="#ff7f0e", linestyle="--", linewidth=1.4)

    if events is not None:
        for ed in events.index:
            if real_prices.index.min() <= ed <= real_prices.index.max():
                ax1.axvline(ed, color="red", alpha=0.35, linewidth=0.9, linestyle=":")

        # Légende choc (une seule entrée)
        ax1.axvline(pd.Timestamp("1900-01-01"), color="red", alpha=0.35,
                    linewidth=0.9, linestyle=":", label="Choc tarifaire")

    ax1.set_title(title, fontsize=13, fontweight="bold")
    ax1.set_ylabel("Prix (€)")
    ax1.legend(framealpha=0.9)
    ax1.grid(True, alpha=0.3)

    # Panneau inférieur : erreur absolue
    err = (real_prices - pred_prices).abs()
    ax2.fill_between(err.index, err.values, alpha=0.5, color="#d62728")
    ax2.set_ylabel("Erreur absolue")
    ax2.set_xlabel("Date")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"  Figure sauvegardée : {save_path}")
    plt.show()


def plot_returns_comparison(
    real_returns: pd.Series,
    pred_returns: pd.Series,
    events: Optional[pd.DataFrame] = None,
    title: str = "ARIMA — Rendements réels vs prédits",
    save_path: Optional[str] = None,
) -> None:
    """Graphique des log-rendements avec zones de chocs surlignées."""
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(real_returns.index, real_returns.values,
            label="Rendements réels", color="#1f77b4", alpha=0.8, linewidth=1)
    ax.plot(pred_returns.index, pred_returns.values,
            label="ARIMA prédit", color="#ff7f0e", linestyle="--", alpha=0.9, linewidth=1)

    if events is not None:
        for ed in events.index:
            if real_returns.index.min() <= ed <= real_returns.index.max():
                ax.axvspan(
                    ed - pd.Timedelta(days=5),
                    ed + pd.Timedelta(days=5),
                    alpha=0.12, color="red"
                )

    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_title(title, fontsize=12)
    ax.set_ylabel("Log-rendement")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.show()


# ──────────────────────────────────────────────────────────────────────────────
# VIII. PIPELINE PRINCIPAL
# ──────────────────────────────────────────────────────────────────────────────

def run_arima_pipeline(
    filepath: str,
    date_col: str = "date",
    close_col: str = "close",
    train_start: str = "2018-01-02",
    train_end: str = "2024-12-31",
    test_start: str = "2025-01-01",
    test_end: str = "2025-11-18",
    events_filepath: Optional[str] = None,
    use_arimax: bool = True,
    label: str = "Asset",
    save_dir: Optional[str] = None,
) -> Dict:
    """
    Pipeline complet ARIMA / ARIMAX.

    Étapes
    ------
    1. Chargement des données et construction des log-rendements
    2. Construction des tariff features (si events_filepath fourni)
    3. Test ADF de stationnarité
    4. Découpage train / test par dates
    5. Sélection automatique de l'ordre (p, d, q)
    6. Walk-forward J+1 optimisé
    7. Reconstruction des prix
    8. Calcul des métriques (RMSE, MAE, DA, Event-Window Accuracy)
    9. Visualisation et sauvegarde optionnelle

    Retourne
    --------
    dict avec : order, metrics_returns, metrics_prices, predictions DataFrame
    """
    print(f"\n{'='*60}")
    print(f"  ARIMA Pipeline — {label}")
    print(f"{'='*60}")

    # ── 1. Données ──
    exog_cols = None
    events = None

    if events_filepath and os.path.exists(events_filepath) and use_arimax:
        events = load_tariff_events(events_filepath)

    log_returns, exog_all = load_and_prepare(
        filepath, date_col, close_col, exog_cols=exog_cols
    )

    # Construction des tariff features comme exogènes si events disponibles
    if events is not None and use_arimax:
        tariff_feat = build_tariff_features(log_returns.index, events)
        exog_all = tariff_feat
        print(f"  ARIMAX activé — {len(tariff_feat.columns)} features tarifaires")
    else:
        print("  Mode ARIMA pur (pas de features exogènes)")

    # ── 2. ADF ──
    adf_test(log_returns, label=label)

    # ── 3. Découpage ──
    train_ret = log_returns.loc[train_start:train_end]
    test_ret  = log_returns.loc[test_start:test_end]
    print(f"  Train : {train_ret.index.min().date()} → {train_ret.index.max().date()} ({len(train_ret)} obs)")
    print(f"  Test  : {test_ret.index.min().date()} → {test_ret.index.max().date()} ({len(test_ret)} obs)")

    exog_train = exog_all.loc[train_ret.index] if exog_all is not None else None
    exog_full  = exog_all if exog_all is not None else None

    # ── 4. Sélection de l'ordre ──
    print("\n  Sélection de l'ordre ARIMA :")
    order = select_order(train_ret, exog_train=exog_train)

    # ── 5. Walk-forward ──
    print(f"\n  Walk-forward ({len(test_ret)} pas)...")
    pred_returns = walk_forward_optimized(
        log_returns, test_ret.index, order, full_exog=exog_full
    )

    # ── 6. Prix ──
    prices_series = np.exp(np.log(
        pd.read_csv(filepath).pipe(lambda d: (
            d.rename(columns={c: c.lower() for c in d.columns})
        ))
    ).head(1))  # Reconstruction propre via le CSV

    # Approche plus robuste : relire directement le CSV pour le prix anchor
    df_raw = pd.read_csv(filepath)
    df_raw.columns = [c.lower().strip() for c in df_raw.columns]
    df_raw["date"] = pd.to_datetime(df_raw[date_col.lower()])
    df_raw = df_raw.set_index("date").sort_index()
    df_raw.index = df_raw.index.tz_localize(None) if df_raw.index.tz else df_raw.index

    prices_raw = df_raw[close_col.lower()].astype(float)
    anchor = float(prices_raw.loc[prices_raw.index <= pd.Timestamp(train_end)].iloc[-1])
    pred_prices = reconstruct_prices(anchor, pred_returns)

    real_prices = prices_raw.loc[test_ret.index]
    real_prices = real_prices.reindex(pred_prices.index, method="ffill")

    # ── 7. Métriques ──
    events_idx = events.index if events is not None else None
    metrics_ret = compute_metrics(
        test_ret, pred_returns, label=f"{label} (rendements)", events_index=events_idx
    )
    metrics_px = compute_metrics(
        real_prices, pred_prices, label=f"{label} (prix)"
    )

    # ── 8. Sauvegarde résultats ──
    results_df = pd.DataFrame({
        "real_returns":  test_ret,
        "pred_returns":  pred_returns,
        "real_prices":   real_prices,
        "pred_prices":   pred_prices,
    })

    output = {
        "label":           label,
        "order":           order,
        "metrics_returns": metrics_ret,
        "metrics_prices":  metrics_px,
        "predictions":     results_df,
    }

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        csv_path = os.path.join(save_dir, f"arima_{label.lower()}_predictions.csv")
        results_df.to_csv(csv_path)
        print(f"\n  Prédictions sauvegardées : {csv_path}")

    # ── 9. Visualisation ──
    fig_path = os.path.join(save_dir, f"arima_{label.lower()}_prices.png") if save_dir else None
    plot_predictions(real_prices, pred_prices, events=events,
                     title=f"ARIMA — {label} | Prix réels vs prédits",
                     save_path=fig_path)

    plot_returns_comparison(test_ret, pred_returns, events=events,
                            title=f"ARIMA — {label} | Log-rendements")

    return output


# ──────────────────────────────────────────────────────────────────────────────
# IX. COMPARAISON MULTI-ACTIFS
# ──────────────────────────────────────────────────────────────────────────────

def run_multi_asset(
    assets: Dict[str, str],          # {"Airbus": "data/raw/Airbus.csv", ...}
    events_filepath: Optional[str] = None,
    save_dir: str = "data/reports",
    **kwargs,
) -> pd.DataFrame:
    """
    Lance le pipeline sur plusieurs actifs et produit un tableau comparatif.

    Retourne un DataFrame avec une ligne par actif et les métriques en colonnes.
    """
    rows = []
    for name, path in assets.items():
        result = run_arima_pipeline(
            filepath=path,
            label=name,
            events_filepath=events_filepath,
            save_dir=save_dir,
            **kwargs,
        )
        row = {"Asset": name, "ARIMA_order": str(result["order"])}
        row.update({f"ret_{k}": v for k, v in result["metrics_returns"].items()})
        row.update({f"px_{k}":  v for k, v in result["metrics_prices"].items()})
        rows.append(row)

    comparison = pd.DataFrame(rows).set_index("Asset")
    print("\n" + "="*60)
    print("  TABLEAU COMPARATIF ARIMA — TOUS ACTIFS")
    print("="*60)
    print(comparison.to_string())

    os.makedirs(save_dir, exist_ok=True)
    comparison.to_csv(os.path.join(save_dir, "arima_comparison.csv"))
    return comparison


# ──────────────────────────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    ASSETS = {
        "Airbus":    "data/raw/Airbus.csv",
        "LVMH":      "data/raw/LVMH.csv",
        "Stellantis":"data/raw/Stellantis.csv",
    }

    comparison_df = run_multi_asset(
        assets=ASSETS,
        events_filepath="data/events/tariff_events.csv",  # créer ce fichier
        save_dir="data/reports",
        train_start="2018-01-02",
        train_end="2024-12-31",
        test_start="2025-01-01",
        test_end="2025-11-18",
        use_arimax=True,
    )