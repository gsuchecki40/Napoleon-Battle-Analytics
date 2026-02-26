# napoleon-battle-analytics

> Clustering 660 battles across 400 years of warfare to quantify military genius — Napoleon, Wellington, Frederick the Great, and more.

## Live Dashboard

**[Live Dashboard](https://gsuchecki40.github.io/napoleon-battle-analytics/)**

Interactive dashboard with animated general comparisons, cluster breakdowns, and Monte Carlo head-to-head matchups. Worth clicking.

---

## Overview

This project applies unsupervised machine learning to the CDB90 military battle database — 660 land battles fought between 1600 and 1973. The goal was to identify structural archetypes in how battles are fought and use those archetypes to compare historical commanders across eras.

The core question: was Napoleon's greatness about genius, or about preparation?

The data suggests it was mostly preparation. He fought outnumbered in only 12% of his battles — the lowest of any general analyzed. He engineered favorable conditions before the first shot was fired.

---

## Data Source

**CDB90 Battle Database** — originally compiled by the U.S. Army Concepts Analysis Agency (1990), cleaned and packaged by [jrnold/CDB90](https://github.com/jrnold/CDB90).

Features include army strengths, casualties, commanders, tactical schemes, terrain, weather, front widths, and battle duration across 660 engagements.

---

## Pipeline

```
CDB90/data/
    battles.csv
    belligerents.csv
    battle_durations.csv
    front_widths.csv
    terrain.csv
    weather.csv
    weather.csv
```

**1. Data Loading & Joining** (`load_cdb90.py`)
Loads all CDB90 tables and pivots belligerents into attacker/defender columns, joining on `isqno`.

**2. Feature Engineering** (`battledata.py`)
- Force ratio, casualty loss percentages, exchange ratio
- Casualty intensity (total casualties / total troops)
- Attacker underdog flag (force ratio < 0.8)
- Log transforms on skewed size/casualty columns
- 99th percentile clipping for outlier control

**3. Clustering** (`battleclusters.py`)
- StandardScaler normalization
- PCA reduction to 10 components
- K-Means (k=8) — chosen via silhouette scoring
- HDBSCAN for density-based comparison
- UMAP 2D projection for visualization

**4. General Comparison** (`napoleon_stats.py`)
Filters belligerents by commander name, joins cluster labels and engineered features, computes win rate, avg achievement score, casualty intensity, and underdog rate per general.

**5. Monte Carlo Simulation** (`headtohead_montecarlo.py`)
For each matchup, samples 100,000 achievement scores from each general's empirical distribution and counts wins. Results broken out by shared cluster type.

**6. Dashboard** (`index.html`)
Standalone HTML/CSS/JS dashboard. No dependencies. Animated bars, tabbed metric comparison, cluster cards, and head-to-head matchup visualization.

---

## Cluster Archetypes

| # | Name | Description |
|---|------|-------------|
| 0 | Large-Scale Attritional | Large armies, moderate casualties, attacker edges out a win |
| 1 | High-Intensity Defensive | Bloody, defender dominates |
| 2 | Decisive Pursuit | Mobile, low casualties, attacker wins cleanly |
| 3 | Small-Scale Engagement | Tiny armies, minimal casualties |
| 4 | High-Intensity Offensive | Small armies, highest intensity, attacker annihilates |
| 5 | Massive Set-Piece | Enormous armies, both sides committed, epic clashes |
| 6 | Failed Assault | Attacker probes, finds resistance, pulls back |
| 7 | Operational-Scale Annihilation | WWI/WWII army-group scale, industrial casualties |

---

## Key Findings

**Napoleon (25 battles, 80% win rate)**
Dominated Large-Scale Attritional battles (83% win rate). Only fought outnumbered 12% of the time — lowest in the dataset. His genius was manufacturing favorable conditions, not overcoming impossible odds.

**Wellington (6 battles, 100% win rate)**
Never lost. Fought outnumbered 33% of the time. Kept casualty intensity low. Monte Carlo simulation gives him a 51.4% edge over Napoleon head-to-head.

**Frederick the Great (14 battles, 78.6% win rate)**
Highest raw achievement score (7.43) in the dataset. Fought outnumbered 29% of the time. The most underrated profile in the analysis.

**Robert E. Lee (12 battles, 50% win rate)**
Never fought as an underdog by force ratio. Still only won half his battles. The data is not kind to the mythology.

**Grant (8 battles, 62.5% win rate)**
Fought outnumbered 37.5% of the time — most of any general analyzed. Still won nearly two-thirds of his battles.

---

## Head-to-Head Results (Monte Carlo, 100k simulations)

| Matchup | Winner | Probability |
|---------|--------|-------------|
| Napoleon vs Wellington | Wellington | 51.4% – 48.6% |
| Napoleon vs Lee | Napoleon | 52.7% – 47.3% |
| Napoleon vs Jackson | Napoleon | 67.0% – 33.0% |
| Grant vs Lee | Lee | 60.4% – 39.6% |

---

## Stack

- Python 3.11
- pandas, numpy, scikit-learn
- umap-learn, hdbscan
- matplotlib, seaborn

---

## Usage

```bash
git clone https://github.com/yourusername/napoleon-battle-analytics
cd napoleon-battle-analytics
git clone https://github.com/jrnold/CDB90.git
pip install pandas numpy scikit-learn umap-learn hdbscan matplotlib seaborn

python battledata.py          # build feature matrix
python battleclusters.py      # run clustering
python napoleon_stats.py      # general comparison
python headtohead_montecarlo.py  # simulations
```

Open `index.html` in a browser for the full dashboard.
