import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from scipy import stats

#Load
bel = pd.read_csv('./BattleML/CDB90/data/belligerents.csv')
df  = pd.read_csv('./BattleML/data/battles_clustered.csv')

bel['co_clean'] = bel['co'].replace({
    'BONAPARTE':             'NAPOLEON I',
    'WELLINGTON & BLUECHER': 'WELLINGTON',
})

bel_merged = bel.merge(df[['isqno', 'kmeans', 'casualty_intensity',
                             'force_ratio', 'attacker_underdog']], on='isqno', how='left')

cluster_names = {
    0: "Large-Scale Attritional",
    1: "High-Intensity Defensive",
    2: "Decisive Pursuit",
    3: "Small-Scale Engagement",
    4: "High-Intensity Offensive",
    5: "Massive Set-Piece",
    6: "Failed Assault",
    7: "Operational-Scale Annihilation",
}

N_SIMS = 100_000

MATCHUPS = [
    ("NAPOLEON I", "WELLINGTON"),
    ("NAPOLEON I", "LEE"),
    ("NAPOLEON I", "JACKSON"),
    ("GRANT",      "LEE")
]

#Monte Carlo

def get_ach_by_cluster(general):
    rows = bel_merged[bel_merged['co_clean'] == general][['ach', 'kmeans']].dropna()
    return rows

def monte_carlo(gen_a, gen_b, n_sims=N_SIMS, seed=42):
    rng = np.random.default_rng(seed)

    data_a = get_ach_by_cluster(gen_a)
    data_b = get_ach_by_cluster(gen_b)

    # Find shared cluster types
    clusters_a = set(data_a['kmeans'].unique())
    clusters_b = set(data_b['kmeans'].unique())
    shared = clusters_a & clusters_b

    if not shared:
        # No shared clusters — use full ach distributions
        ach_a = data_a['ach'].values
        ach_b = data_b['ach'].values
        cluster_label = "All Clusters (no overlap)"
        results = {'Overall': _run_sims(ach_a, ach_b, n_sims, rng)}
        return results, cluster_label

    # Run sims per shared cluster, weighted by combined battle count
    results = {}
    weights = {}
    for c in sorted(shared):
        ach_a = data_a[data_a['kmeans'] == c]['ach'].values
        ach_b = data_b[data_b['kmeans'] == c]['ach'].values
        if len(ach_a) < 2 or len(ach_b) < 2:
            continue
        results[cluster_names[c]] = _run_sims(ach_a, ach_b, n_sims, rng)
        weights[cluster_names[c]] = len(ach_a) + len(ach_b)

    # Overall weighted win probability
    if weights:
        total = sum(weights.values())
        overall_win_a = sum(
            results[k]['win_pct_a'] * weights[k] / total
            for k in weights
        )
        results['OVERALL'] = {
            'win_pct_a': overall_win_a,
            'win_pct_b': 100 - overall_win_a,
            'draw_pct':  0.0,
            'mean_a':    data_a['ach'].mean(),
            'mean_b':    data_b['ach'].mean(),
            'n_a':       len(data_a),
            'n_b':       len(data_b),
        }

    return results

def _run_sims(ach_a, ach_b, n_sims, rng):
    sims_a = rng.choice(ach_a, size=n_sims, replace=True)
    sims_b = rng.choice(ach_b, size=n_sims, replace=True)
    wins_a = np.sum(sims_a > sims_b)
    wins_b = np.sum(sims_b > sims_a)
    draws  = n_sims - wins_a - wins_b
    return {
        'win_pct_a': wins_a / n_sims * 100,
        'win_pct_b': wins_b / n_sims * 100,
        'draw_pct':  draws  / n_sims * 100,
        'mean_a':    ach_a.mean(),
        'mean_b':    ach_b.mean(),
        'n_a':       len(ach_a),
        'n_b':       len(ach_b),
    }



# RUN ALL MATCHUPS

all_results = {}
for gen_a, gen_b in MATCHUPS:
    print(f"\n{'='*55}")
    print(f"  {gen_a}  vs  {gen_b}")
    print(f"{'='*55}")
    results = monte_carlo(gen_a, gen_b)
    all_results[(gen_a, gen_b)] = results

    for context, r in results.items():
        print(f"  [{context}]")
        print(f"    {gen_a:20s}  win: {r['win_pct_a']:5.1f}%  |  avg ach: {r['mean_a']:.2f}  (n={r['n_a']})")
        print(f"    {gen_b:20s}  win: {r['win_pct_b']:5.1f}%  |  avg ach: {r['mean_b']:.2f}  (n={r['n_b']})")
        print(f"    Draw:                   {r['draw_pct']:5.1f}%")



#Viz

fig, axes = plt.subplots(len(MATCHUPS), 1, figsize=(14, 4 * len(MATCHUPS)))
fig.suptitle(f'Head-to-Head Monte Carlo Simulations  (n={N_SIMS:,} each)',
             fontsize=14, fontweight='bold', y=1.01)

COLOR_A    = "#C0392B"   # red  — left general
COLOR_B    = "#2980B9"   # blue — right general
COLOR_DRAW = "#BDC3C7"   # grey — draw

for ax, (gen_a, gen_b) in zip(axes, MATCHUPS):
    results = all_results[(gen_a, gen_b)]
    contexts = list(results.keys())
    y_pos = np.arange(len(contexts))

    wins_a  = [results[c]['win_pct_a'] for c in contexts]
    draws   = [results[c]['draw_pct']  for c in contexts]
    wins_b  = [results[c]['win_pct_b'] for c in contexts]

    # Stacked horizontal bars
    bars_a = ax.barh(y_pos, wins_a, color=COLOR_A, edgecolor='none', alpha=0.85)
    bars_d = ax.barh(y_pos, draws,  left=wins_a, color=COLOR_DRAW, edgecolor='none', alpha=0.6)
    bars_b = ax.barh(y_pos, wins_b, left=[a+d for a, d in zip(wins_a, draws)],
                     color=COLOR_B, edgecolor='none', alpha=0.85)

    # Labels inside bars
    for i, (wa, dr, wb) in enumerate(zip(wins_a, draws, wins_b)):
        if wa > 6:
            ax.text(wa/2, i, f'{wa:.1f}%', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')
        if wb > 6:
            ax.text(100 - wb/2, i, f'{wb:.1f}%', ha='center', va='center',
                    fontsize=8, color='white', fontweight='bold')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(contexts, fontsize=8.5)
    ax.set_xlim(0, 100)
    ax.axvline(50, color='black', linewidth=0.8, linestyle='--', alpha=0.4)
    ax.set_xlabel('Win Probability %', fontsize=9)
    ax.set_facecolor('#F7F7F7')
    ax.spines[['top', 'right']].set_visible(False)

    patch_a    = mpatches.Patch(color=COLOR_A,    label=gen_a)
    patch_b    = mpatches.Patch(color=COLOR_B,    label=gen_b)
    patch_draw = mpatches.Patch(color=COLOR_DRAW, label='Draw')
    ax.legend(handles=[patch_a, patch_draw, patch_b],
              loc='lower right', fontsize=8, framealpha=0.9)
    ax.set_title(f'{gen_a}  vs  {gen_b}', fontsize=11, fontweight='bold', pad=6)

fig.tight_layout()
fig.savefig('./BattleML/data/headtohead_montecarlo.png', dpi=200, bbox_inches='tight')
plt.close()
print("\nSaved: headtohead_montecarlo.png")
