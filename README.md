# napoleon-battle-analytics

> Clustering 660 battles across 400 years of warfare to quantify military genius — Napoleon, Wellington, Frederick the Great, and more.

## Live Dashboard

**[Live Dashboard](https://gsuchecki40.github.io/Napoleon-Battle-Analytics/)**

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

## Feature Definitions

### Raw CDB90 Fields

| Feature | Description |
|---------|-------------|
| `str` | Initial troop strength at the start of the engagement, in number of personnel |
| `cas` | Total casualties sustained — killed, wounded, captured, and missing |
| `finst` | Final strength at the end of the engagement after casualties |
| `ach` | Achievement score (0–10) assigned by CDB90 analysts reflecting how well a side achieved its tactical objectives. 0 = total failure, 5 = stalemate, 10 = complete success. This is the primary outcome metric used throughout the analysis. |
| `cav` | Number of cavalry units committed |
| `arty` | Number of artillery pieces committed |
| `tank` | Number of tanks committed (zero-filled for pre-WWI engagements where tanks did not exist) |
| `pri1` | Primary tactical scheme code — the main maneuver type employed (e.g. frontal assault, envelopment, defense) |
| `duration1` | Length of the engagement in days |
| `wofa / wofd` | Width of front in kilometers for attacker and defender respectively |
| `terra1` | Primary terrain type code (e.g. open, wooded, urban, mountainous) |
| `wx1` | Primary weather condition code (e.g. clear, rain, snow, fog) |

### Engineered Features

| Feature | Formula | Description |
|---------|---------|-------------|
| `force_ratio` | `att_str / def_str` | Ratio of attacker to defender strength at battle start. Values above 1.0 mean the attacker had more troops. Used to determine whether a general fought as an underdog. |
| `att_loss_pct` | `att_cas / att_str` | Fraction of the attacker's initial force lost as casualties. Clipped at the 99th percentile to handle data quality issues where reported casualties exceeded reported strength. |
| `def_loss_pct` | `def_cas / def_str` | Same as above for the defender. |
| `exchange_ratio` | `att_cas / def_cas` | Ratio of attacker casualties to defender casualties. Values below 1.0 mean the attacker inflicted more casualties than they suffered. Clipped at the 99th percentile. |
| `casualty_intensity` | `(att_cas + def_cas) / (att_str + def_str)` | Total casualties as a fraction of total troops engaged. The primary measure of how bloody a battle was regardless of who won. A value of 0.20 means 20% of all troops on both sides became casualties. |
| `total_troops` | `att_str + def_str` | Combined troop count for both sides. Used as a proxy for battle scale. |
| `attacker_underdog` | `1 if force_ratio < 0.80 else 0` | Binary flag indicating the attacker had fewer than 80% of the defender's troop strength. Used to measure how often a general chose to engage at a numerical disadvantage. |
| `ach_diff` | `att_ach - def_ach` | Difference in achievement scores between attacker and defender. Positive values indicate attacker dominance. Used to assess decisiveness of outcome beyond a binary win/loss. |

### Metric Definitions Used in Analysis

**Win Rate** — percentage of battles where a general's side recorded an achievement score of 6 or higher out of 10. A score of 6 represents a clear tactical success; scores of 5 and below represent stalemates or failures. This threshold was chosen to distinguish meaningful victories from marginal outcomes.

**Average Achievement Score** — mean `ach` score across all of a general's battles (0–10 scale). Captures not just whether a general won, but how decisively. A general who wins 10-0 every time scores higher than one who wins 6-5 every time.

**Casualty Intensity** — mean `casualty_intensity` across a general's battles. Reflects the character of warfare a general engaged in — lower values indicate more efficient or mobile battles, higher values indicate grinding attritional combat.

**Underdog Rate** — percentage of battles where `force_ratio < 0.80`, meaning the general's side had fewer than 80% of the enemy's troop strength at the start. A measure of how often a general chose to fight at a numerical disadvantage.

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
```

**1. Data Loading & Joining** (`load_cdb90.py`)
Loads all CDB90 tables and pivots belligerents into attacker/defender columns, joining on `isqno`.

**2. Feature Engineering** (`battledata.py`)
Constructs all engineered features listed above. Applies 99th percentile clipping to ratio-based features to handle records where reported casualties exceeded reported strength. Log transforms applied to `att_str`, `def_str`, `att_cas`, `def_cas`, `total_troops`, `exchange_ratio`, `force_ratio`, and `duration1` to reduce right skew before clustering. Tanks imputed to zero for all pre-WWI battles.

**3. Clustering** (`battleclusters.py`)
- StandardScaler normalization across all features
- PCA reduction to 10 components
- K-Means (k=8) — chosen via silhouette scoring
- HDBSCAN for density-based comparison
- UMAP 2D projection for visualization

**4. General Comparison** (`napoleon_stats.py`)
Filters belligerents by commander name, joins cluster labels and engineered features, computes win rate, avg achievement score, casualty intensity, and underdog rate per general.

**5. Monte Carlo Simulation** (`headtohead_montecarlo.py`)
For each matchup, samples 100,000 achievement scores from each general's empirical distribution and counts wins. Results broken out by shared cluster type. When two generals share no cluster types, the simulation runs on full career distributions.

**6. Dashboard** (`index.html`)
Standalone HTML/CSS/JS dashboard. No dependencies. Animated bars, tabbed metric comparison, cluster cards, and head-to-head matchup visualization.

---

## Cluster Archetypes

| # | Name | Median Intensity | Typical Outcome | Description |
|---|------|-----------------|-----------------|-------------|
| 0 | Large-Scale Attritional | 0.119 | Attacker wins narrowly | Long engagements between large forces, moderate casualties, grinding rather than decisive |
| 1 | High-Intensity Defensive | 0.219 | Defender wins decisively | Fortified positions, river crossings, failed offensives — defender holds and inflicts heavy losses |
| 2 | Decisive Pursuit | 0.074 | Attacker wins cleanly | Mobile operations, exploitation, flanking — attacker has size advantage and wins with low casualties |
| 3 | Small-Scale Engagement | 0.043 | Slight attacker edge | Colonial skirmishes, rearguard actions, minor engagements with minimal casualties on both sides |
| 4 | High-Intensity Offensive | 0.295 | Attacker annihilates | Short, brutal, decisive — attacker crushes defender in under two days with heavy losses on both sides |
| 5 | Massive Set-Piece | 0.223 | Near-even outcome | The great clashes of history — Borodino, Waterloo, Gettysburg — massive armies, both sides committed fully |
| 6 | Failed Assault | 0.049 | Defender wins | Attacker probes with size advantage, finds strong resistance, withdraws with minimal casualties |
| 7 | Operational-Scale Annihilation | 0.326 | Attacker wins decisively | WWI/WWII army-group engagements — Kursk, Brusilov, Moscow — industrial-scale casualties, attacker breaks through |

---

## Key Findings

**Napoleon (25 battles, 80% win rate)**
Dominated Large-Scale Attritional battles (83% win rate, 12 battles). Only fought outnumbered 12% of the time — lowest in the dataset. His genius was manufacturing favorable conditions, not overcoming impossible odds.

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
git clone https://github.com/gsuchecki40/napoleon-battle-analytics
cd napoleon-battle-analytics
git clone https://github.com/jrnold/CDB90.git
pip install pandas numpy scikit-learn umap-learn hdbscan matplotlib seaborn

python battledata.py             # build feature matrix
python battleclusters.py         # run clustering
python napoleon_stats.py         # general comparison
python headtohead_montecarlo.py  # simulations
```

Open `index.html` in a browser for the full dashboard.
