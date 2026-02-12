"""
Pipeline complet de modélisation ARIMA pour la prévision de prix d'actions.

Ce script est conçu pour un usage académique :
- lecture d'un fichier CSV depuis data/raw/
- transformation en log-rendements
- test de stationnarité (ADF)
- sélection automatique des paramètres ARIMA (p,d,q)
- entraînement et prévision
- reconstruction des prix
- calcul des métriques (MAE, RMSE, MAPE) sur rendements et prix
- visualisation des prix réels vs prédits
"""

from __future__ import annotations

import os
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error

from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.stattools import adfuller

# Tentative d'import d'auto_arima (optionnel). Si indisponible, on fera une recherche AIC.
try:
    from pmdarima import auto_arima

    PMDARIMA_AVAILABLE = True
except Exception:
    PMDARIMA_AVAILABLE = False


# --------------------------------------------------------------------------------------
# Fonctions utilitaires
# --------------------------------------------------------------------------------------

def load_csv(filepath: str) -> pd.DataFrame:
    """
    Charge un fichier CSV et renvoie un DataFrame.

    Paramètres
    ----------
    filepath : str
        Chemin du fichier CSV
    ------
    pd.DataFrame
        Données brutes.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Fichier introuvable : {filepath}")
    return pd.read_csv(filepath)


def print_dataset_details(df: pd.DataFrame, label: str = "Dataset") -> None:
    """
    Affiche dans la console des informations clés sur le dataset.

    Paramètres
    ----------
    df : pd.DataFrame
        Données brutes ou indexées.
    label : str
        Libellé affiché dans la console.
    """
    print(f"--- {label} ---")
    print(f"Nombre de lignes : {len(df)}")
    print(f"Nombre de colonnes : {df.shape[1]}")
    print("Colonnes :", list(df.columns))
    print("Types de colonnes :")
    print(df.dtypes)

    # Si l'index est temporel, on affiche la plage de dates
    if isinstance(df.index, pd.DatetimeIndex) and len(df.index) > 0:
        print(f"Plage de dates : {df.index.min().date()} -> {df.index.max().date()}")
        if df.index.freq is not None:
            print(f"Fréquence : {df.index.freqstr}")

    print()


def _resolve_column(df: pd.DataFrame, candidate: str, aliases: Tuple[str, ...]) -> str:
    """
    Paramètres
    ----------
    df : pd.DataFrame
        Données brutes.
    candidate : str
        Nom de colonne proposé.
    aliases : tuple
        Alias acceptés (ex. "date", "Date", "DATE").

    Retour
    ------
    str
        Nom réel de la colonne dans le DataFrame.
    """
    if candidate in df.columns:
        return candidate

    lower_map = {c.lower(): c for c in df.columns}
    if candidate.lower() in lower_map:
        return lower_map[candidate.lower()]

    for alias in aliases:
        if alias in df.columns:
            return alias
        if alias.lower() in lower_map:
            return lower_map[alias.lower()]

    raise ValueError(f"Colonne introuvable : {candidate}. Colonnes disponibles : {list(df.columns)}")


def set_datetime_index(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """
    Convertit la colonne de dates en index temporel et trie les données.
    Paramètres
    ----------
    df : pd.DataFrame
        Données brutes.
    date_col : str
        Nom de la colonne contenant les dates.

    Retour
    ------
    pd.DataFrame
        Données indexées par dates.
    """
    df = df.copy()
    date_col = _resolve_column(df, date_col, aliases=("date", "Date", "DATE"))
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).sort_values(date_col)
    df = df.set_index(date_col)
    return df


def extract_close_prices(df: pd.DataFrame, close_col: str) -> pd.Series:
    """
    Extrait la série de prix de clôture.
    Paramètres
    ----------
    df : pd.DataFrame
        Données indexées par dates.
    close_col : str
        Nom de la colonne des prix de clôture.

    Retour
    ------
    pd.Series
        Série temporelle des prix de clôture.
    """
    close_col = _resolve_column(
        df,
        close_col,
        aliases=("close", "Close", "CLOSE", "Adj Close", "Adj_Close", "adj close", "prix", "Prix"),
    )
    return df[close_col].astype(float)


def compute_log_returns(prices: pd.Series) -> pd.Series:
    """
    Calcule les log-rendements à partir des prix.

    r_t = log(P_t) - log(P_{t-1})

    Paramètres
    ----------
    prices : pd.Series
        Série des prix.

    Retour
    ------
    pd.Series
        Série des log-rendements.
    """
    log_prices = np.log(prices)
    log_returns = log_prices.diff().dropna()
    return log_returns


def regularize_prices_frequency(
    prices: pd.Series,
    preferred_freq: str = "B",
) -> Tuple[pd.Series, Optional[str]]:
    """
    Force une fréquence régulière sur les prix pour éviter les warnings de statsmodels.

    - Si la fréquence est détectable, on la conserve.
    - Sinon, on passe en fréquence 'B' (jours ouvrés) et on propage la dernière
      valeur connue (forward-fill). Cette approximation est courante pour obtenir
      une série régulière, mais elle ajoute des jours sans transaction (retours nuls).

    Paramètres
    ----------
    prices : pd.Series
        Série des prix indexée par dates.
    preferred_freq : str
        Fréquence de repli (par défaut 'B' pour jours ouvrés).

    Retour
    ------
    (pd.Series, Optional[str])
        Série régularisée et fréquence utilisée.
    """
    inferred = pd.infer_freq(prices.index)
    if inferred is not None:
        return prices.asfreq(inferred), inferred

    # Fréquence non détectée : repli sur une fréquence régulière
    return prices.asfreq(preferred_freq, method="ffill"), preferred_freq


def adf_stationarity_test(series: pd.Series) -> Dict[str, float]:
    """
    Effectue un test de Dickey-Fuller augmenté (ADF) pour la stationnarité.

    Paramètres
    ----------
    series : pd.Series
        Série temporelle à tester.

    Retour
    ------
    dict
        Statistique ADF, p-value, valeurs critiques.
    """
    result = adfuller(series, autolag="AIC")
    return {
        "adf_stat": result[0],
        "p_value": result[1],
        "crit_1%": result[4].get("1%"),
        "crit_5%": result[4].get("5%"),
        "crit_10%": result[4].get("10%"),
    }


def train_test_split_series(series: pd.Series, test_size: float = 0.2) -> Tuple[pd.Series, pd.Series]:
    """
    Découpe une série temporelle en train/test sans mélange.

    Paramètres
    ----------
    series : pd.Series
        Série temporelle.
    test_size : float
        Proportion de l'échantillon allouée au test.

    Retour
    ------
    (pd.Series, pd.Series)
        Séries train et test.
    """
    split_idx = int(len(series) * (1 - test_size))
    train = series.iloc[:split_idx]
    test = series.iloc[split_idx:]
    return train, test


def train_test_split_by_date(
    series: pd.Series,
    train_start: str,
    train_end: str,
    test_start: str,
    test_end: str,
) -> Tuple[pd.Series, pd.Series]:
    """
    Découpe une série temporelle en train/test selon des bornes de dates explicites (inclusives).

    Paramètres
    ----------
    series : pd.Series
        Série temporelle avec index datetime.
    train_start, train_end, test_start, test_end : str
        Bornes temporelles au format YYYY-MM-DD.

    Retour
    ------
    (pd.Series, pd.Series)
        Séries train et test.
    """
    train = series.loc[train_start:train_end]
    test = series.loc[test_start:test_end]

    if train.empty:
        raise ValueError("Découpage train vide : vérifiez les bornes de dates.")
    if test.empty:
        raise ValueError("Découpage test vide : vérifiez les bornes de dates.")

    return train, test


def select_arima_order(
    series: pd.Series,
    max_p: int = 3,
    max_d: int = 1,
    max_q: int = 3,
) -> Tuple[int, int, int]:
    """
    Sélectionne (p, d, q) automatiquement.

    - Si pmdarima est disponible : auto_arima.
    - Sinon : recherche par grille avec critère AIC.

    Paramètres
    ----------
    series : pd.Series
        Série temporelle stationnaire (log-rendements).
    max_p, max_d, max_q : int
        Bornes de la grille de recherche.

    Retour
    ------
    (int, int, int)
        Ordre ARIMA optimal.
    """
    if PMDARIMA_AVAILABLE:
        model = auto_arima(
            series,
            start_p=0,
            start_q=0,
            max_p=max_p,
            max_q=max_q,
            d=None,
            max_d=max_d,
            seasonal=False,
            trace=False,
            error_action="ignore",
            suppress_warnings=True,
            stepwise=True,
        )
        order = model.order
        return order

    # Recherche manuelle avec AIC
    best_aic = np.inf
    best_order = (0, 0, 0)

    for p in range(0, max_p + 1):
        for d in range(0, max_d + 1):
            for q in range(0, max_q + 1):
                try:
                    model = ARIMA(series, order=(p, d, q))
                    fitted = model.fit()
                    if fitted.aic < best_aic:
                        best_aic = fitted.aic
                        best_order = (p, d, q)
                except Exception:
                    # Certains paramètres peuvent être non valides
                    continue

    return best_order


def train_arima(series: pd.Series, order: Tuple[int, int, int]):
    """
    Entraîne un modèle ARIMA.

    Paramètres
    ----------
    series : pd.Series
        Série d'entraînement.
    order : (int, int, int)
        Ordre (p,d,q).

    Retour
    ------
    ARIMAResults
        Modèle ajusté.
    """
    model = ARIMA(series, order=order)
    return model.fit()


def forecast_returns(model, steps: int) -> pd.Series:
    """
    Fait des prévisions de log-rendements.

    Paramètres
    ----------
    model : ARIMAResults
        Modèle ajusté.
    steps : int
        Nombre de pas de prévision.

    Retour
    ------
    pd.Series
        Log-rendements prédits.
    """
    return model.forecast(steps=steps)


def walk_forward_forecast(
    full_returns: pd.Series,
    test_index: pd.DatetimeIndex,
    order: Tuple[int, int, int],
) -> pd.Series:
    """
    Prévision en walk-forward J+1 (expanding window).

    Pour chaque date t du test :
    - on ajuste le modèle sur les données <= t-1
    - on prédit r_t
    """
    preds = []
    for t in test_index:
        history = full_returns.loc[:t].iloc[:-1]
        try:
            model = ARIMA(history, order=order)
            fitted = model.fit()
            pred = fitted.forecast(steps=1).iloc[0]
        except Exception:
            pred = 0.0
        preds.append(pred)

    return pd.Series(preds, index=test_index)


def reconstruct_prices(
    last_train_price: float,
    predicted_returns: pd.Series,
) -> pd.Series:
    """
    Reconstruit les prix à partir des log-rendements prédits.

    P_t = P_{t-1} * exp(r_t)

    Paramètres
    ----------
    last_train_price : float
        Dernier prix observé dans la partie train (point d'ancrage).
    predicted_returns : pd.Series
        Log-rendements prédits.

    Retour
    ------
    pd.Series
        Série des prix reconstruits.
    """
    prices = []
    last_price = last_train_price

    for r in predicted_returns:
        last_price = last_price * np.exp(r)
        prices.append(last_price)

    return pd.Series(prices, index=predicted_returns.index)


def compute_metrics(y_true: pd.Series, y_pred: pd.Series) -> Dict[str, float]:
    """
    Calcule MAE, RMSE et MAPE.

    Paramètres
    ----------
    y_true : pd.Series
        Valeurs réelles.
    y_pred : pd.Series
        Valeurs prédites.

    Retour
    ------
    dict
        MAE, RMSE, MAPE.
    """
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))

    # MAPE : éviter la division par zéro
    eps = 1e-8
    mape = np.mean(np.abs((y_true - y_pred) / (y_true + eps))) * 100

    return {"MAE": mae, "RMSE": rmse, "MAPE": mape}


def plot_prices(real_prices: pd.Series, predicted_prices: pd.Series, title: str) -> None:
    """
    Affiche un graphique des prix réels vs prédits.

    Paramètres
    ----------
    real_prices : pd.Series
        Prix observés (partie test).
    predicted_prices : pd.Series
        Prix reconstruits à partir des rendements prédits.
    title : str
        Titre du graphique.
    """
    plt.figure(figsize=(12, 6))
    plt.plot(real_prices.index, real_prices.values, label="Prix réels", linewidth=2)
    plt.plot(predicted_prices.index, predicted_prices.values, label="Prix prédits", linestyle="--")
    plt.title(title)
    plt.xlabel("Date")
    plt.ylabel("Prix")
    plt.legend()
    plt.tight_layout()
    plt.show()


# --------------------------------------------------------------------------------------
# Pipeline principal
# --------------------------------------------------------------------------------------

def run_arima_pipeline(
    filepath: str,
    date_col: str = "date",
    close_col: str = "close",
    test_size: float = 0.2,
    train_start: Optional[str] = None,
    train_end: Optional[str] = None,
    test_start: Optional[str] = None,
    test_end: Optional[str] = None,
    label: Optional[str] = None,
) -> None:
    """
    Exécute le pipeline complet ARIMA.

    Paramètres
    ----------
    filepath : str
        Chemin du fichier CSV.
    date_col : str
        Nom de la colonne de dates.
    close_col : str
        Nom de la colonne des prix de clôture.
    test_size : float
        Proportion de l'échantillon allouée au test.
    """
    # 1) Chargement des données
    df = load_csv(filepath)
    label_prefix = f"[{label}] " if label else ""
    print_dataset_details(df, label=f"{label_prefix}Détails du dataset (brut)")

    # 2) Mise en place de l'index temporel
    df = set_datetime_index(df, date_col)
    print_dataset_details(df, label=f"{label_prefix}Détails du dataset (indexé)")

    # 3) Extraction des prix de clôture
    prices = extract_close_prices(df, close_col)

    # 4) Régularisation de la fréquence pour éviter les warnings statsmodels
    prices, freq_used = regularize_prices_frequency(prices, preferred_freq="B")
    if freq_used is not None:
        print(f"{label_prefix}Fréquence utilisée pour la série de prix : {freq_used}")
        print()

    # 5) Transformation en log-rendements
    log_returns = compute_log_returns(prices)

    # 6) Test de stationnarité (ADF)
    adf_results = adf_stationarity_test(log_returns)
    print("--- Test ADF sur les log-rendements ---")
    for k, v in adf_results.items():
        print(f"{k}: {v}")
    print()

    # 7) Découpage train/test
    # Si des bornes de dates sont fournies, on privilégie ce découpage.
    if all([train_start, train_end, test_start, test_end]):
        train_returns, test_returns = train_test_split_by_date(
            log_returns, train_start, train_end, test_start, test_end
        )
        print(f"--- {label_prefix}Découpage train/test (par dates) ---")
        print(f"Train : {train_start} -> {train_end} | {len(train_returns)} points")
        print(f"Test  : {test_start} -> {test_end} | {len(test_returns)} points")
        print()
    else:
        train_returns, test_returns = train_test_split_series(log_returns, test_size=test_size)
        print(f"--- {label_prefix}Découpage train/test (proportion) ---")
        print(f"Test size : {test_size:.0%}")
        print(f"Train : {train_returns.index.min().date()} -> {train_returns.index.max().date()} | {len(train_returns)} points")
        print(f"Test  : {test_returns.index.min().date()} -> {test_returns.index.max().date()} | {len(test_returns)} points")
        print()

    # 8) Sélection automatique des paramètres (p,d,q)
    order = select_arima_order(train_returns)
    print(f"{label_prefix}Ordre ARIMA sélectionné : {order}")

    # 9) Prévisions en walk-forward J+1 sur la partie test
    predicted_returns = walk_forward_forecast(log_returns, test_returns.index, order)

    # 10) Reconstruction des prix à partir des rendements prédits
    last_train_price = prices.loc[train_returns.index[-1]]
    predicted_prices = reconstruct_prices(last_train_price, predicted_returns)

    # Série des prix réels correspondant à la période de test
    real_prices = prices.loc[test_returns.index]

    # 11) Calcul des métriques
    metrics_returns = compute_metrics(test_returns, predicted_returns)
    metrics_prices = compute_metrics(real_prices, predicted_prices)

    print(f"--- {label_prefix}Métriques sur log-rendements ---")
    for k, v in metrics_returns.items():
        print(f"{k}: {v:.6f}")
    print()

    print(f"--- {label_prefix}Métriques sur prix reconstruits ---")
    for k, v in metrics_prices.items():
        print(f"{k}: {v:.6f}")
    print()

    # 12) Visualisation
    plot_title = "Prévision ARIMA - Prix réels vs prédits"
    if label:
        plot_title = f"{plot_title} ({label})"
    plot_prices(real_prices, predicted_prices, title=plot_title)


def main() -> None:
    """
    Point d'entrée principal.

    Modifiez le chemin du fichier et les noms de colonnes si nécessaire.
    """
    # Liste des fichiers à traiter (adapter les noms si besoin)
    files = [
        "Airbus.csv",
        "LVMH.csv",
        "Stellantis.csv",
    ]

    # Adapter si vos colonnes ont d'autres noms
    date_col = "date"
    close_col = "close"

    # Découpage temporel explicite :
    # train : 2018-01-02 à 2024-12-31
    # test  : 2025-01-01 à 2025-11-18
    for fname in files:
        filepath = os.path.join("data", "raw", fname)
        label = os.path.splitext(fname)[0]
        print("=" * 80)
        print(f"Traitement : {label}")
        print("=" * 80)
        run_arima_pipeline(
            filepath,
            date_col=date_col,
            close_col=close_col,
            test_size=0.2,
            train_start="2018-01-02",
            train_end="2024-12-31",
            test_start="2025-01-01",
            test_end="2025-11-18",
            label=label,
        )


if __name__ == "__main__":
    main()
