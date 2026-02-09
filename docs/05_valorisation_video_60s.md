# Plan de Valorisation + Video 60s

## 1. Axe de valorisation choisi

Type: demonstration technique orientee impact decisionnel.

Public cible:
- jury academique,
- partenaires industriels potentiels,
- recruteurs data/IA.

Message central:
"Nous avons transforme une problematique economique en prototype predictif mesurable, documente et testable."

## 2. Supports a produire

1. Poster/slide one-page avec architecture + resultats.
2. Demo live courte (execution pipeline + lecture des rapports).
3. Video 60s (format libre) pour diffusion.

## 3. Storyboard video (60 secondes)

0-10s: Contexte
- "Les annonces tarifaires perturbent les marches."
- Afficher problematique + tickers suivis.

10-25s: Solution
- Montrer pipeline (download -> preprocess -> models -> evaluation).
- Afficher rapidement les fichiers de sortie.

25-40s: Resultats
- Afficher tableau baseline vs enriched.
- Expliquer en 1 phrase: "le modele enrichi ameliore la precision relative (MAPE)."

40-52s: Valeur
- "Prototype fonctionnel TRL5, reproductible, documente."
- Montrer architecture + tests.

52-60s: Ouverture
- "Prochaine etape: enrichir les evenements tarifaires et valider en walk-forward."
- Afficher contact/equipe.

## 4. Script voix off propose (60s)

"Notre projet repond a une question simple: peut-on anticiper l'evolution du cours Airbus
en integrant le contexte de marche et les evenements tarifaires?
Nous avons construit un pipeline complet: collecte des donnees reelles, pretraitement,
entrainement de deux modeles LSTM, puis comparaison automatique des performances.
Le prototype produit des metriques claires, des predictions exportees et une base de decision objective.
Resultat: nous disposons d'une solution fonctionnelle de niveau TRL5, documentee et testee.
La suite consiste a enrichir la qualite des signaux evenementiels pour gagner en robustesse
sur les periodes de forte volatilite."

## 5. Checklist tournage rapide

1. Ecran terminal avec `python3 src/main.py --skip-download`.
2. Capture `data/reports/model_comparison.csv`.
3. Slide architecture simple.
4. Sous-titres et timecodes (respect strict 60s).
