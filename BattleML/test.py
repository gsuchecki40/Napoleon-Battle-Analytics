import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

# ── Load & prep ───────────────────────────────────────────────────────────────
bel = pd.read_csv('./BattleML/CDB90/data/belligerents.csv')
df  = pd.read_csv('./BattleML/data/battles_clustered.csv')

bel['co_clean'] = bel['co'].replace({
    'BONAPARTE':            'NAPOLEON I',
    'WELLINGTON & BLUECHER': 'WELLINGTON',
})

bel_merged = bel.merge(df[['isqno', 'kmeans', 'casualty_intensity',
                             'force_ratio', 'att_ach', 'def_ach',
                             'attacker_underdog']], on='isqno', how='left')

generals = ['NAPOLEON I', 'FREDERICK II', 'LEE', 'WELLINGTON',
            'GRANT', 'ARCHDUKE CHARLES', 'TURENNE', 'JACKSON', 'WASHINGTON']

gen_df = bel_merged[bel_merged['co_clean'].isin(generals)].copy()
gen_df['own_ach'] = gen_df['ach']
gen_df['win'] = (gen_df['own_ach'] >= 6).astype(int)   # ach 6-10 = clear win

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

# Emphasize Napoleon

Napoleon_Color = '#C0392B'
Other_Color = '#95A5A6'

def bar_colors(index, highlight='NAPOLEON I'):
    return [Napoleon_Color if g == highlight else Other_Color for g in index]

# General order: Napoleon first, then sorted by battle count
gen_order = ['NAPOLEON I'] + [g for g in generals if g != 'NAPOLEON I']

win_rate = gen_df.groupby('co_clean')['win'].mean().reindex(gen_order)*100


acheivement_score = gen_df.groupby('co_clean')['own_ach'].mean().reindex(gen_order)
avg_casualty_intensity = gen_df.groupby('co_clean')['casualty_intensity'].mean().reindex(gen_order)

# Fix for undefined variables
avg_ach = acheivement_score
avg_intensity = avg_casualty_intensity

underdog_rate = gen_df.groupby('co_clean')['attacker_underdog'].mean().reindex(gen_order)*100

nap_clusters = (
    gen_df[gen_df['co_clean'] == 'NAPOLEON I']['kmeans']
    .value_counts(normalize=True)
    .reindex(range(8), fill_value=0) * 100
)
others_clusters = (
    gen_df[gen_df['co_clean'] != 'NAPOLEON I']['kmeans']
    .value_counts(normalize=True)
    .reindex(range(8), fill_value=0) * 100
)

# Win rate by cluster for Napoleon only

nap_df = gen_df[gen_df['co_clean'] == 'NAPOLEON I'].copy()
nap_win_by_cluster = nap_df.groupby('kmeans')['win'].agg(['mean','count'])
nap_win_by_cluster['mean'] *= 100
nap_win_by_cluster = nap_win_by_cluster[nap_win_by_cluster['count'] >= 2]  # Only show clusters with 2+ battles for Napoleon
nap_win_by_cluster.index = [cluster_names[i] for i in nap_win_by_cluster.index]

# Plot

fig = plt.figure(figsize=(20, 18))
fig.suptitle("Napoleon's Battle Performace vs Other Great Generals", fontsize=22, fontweight='bold',y=0.98)

gs = fig.add_gridspec(3, 2, hspace=0.55, wspace=0.35)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[0, 1])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1])
ax5 = fig.add_subplot(gs[2, 0])
ax6 = fig.add_subplot(gs[2, 1])

def style_ax(ax):
    ax.set_facecolor('#F7F7F7')
    ax.spines[['top', 'right']].set_visible(False)
    ax.tick_params(axis='x', rotation=30)

def add_value_labels(ax, fmt='{:.1f}'):
    for bar in ax.patches:
        h = bar.get_height()
        if h == 0:
            continue
        ax.text(bar.get_x() + bar.get_width() / 2, h + 0.5,
                fmt.format(h), ha='center', va='bottom', fontsize=7.5)
        
# Win Rate
ax1.bar(gen_order,win_rate,color=bar_colors(gen_order),edgecolor='white')
ax1.set_title('Win Rate (%)', fontsize=14, fontweight='bold')
ax1.set_ylabel('%')
ax1.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0, symbol='%'))
add_value_labels(ax1,'{:.0f}')
style_ax(ax1)

#Avg Acheivement Score
ax2.bar(gen_order, avg_ach, color=bar_colors(gen_order), edgecolor='white')
ax2.set_title('Avg Achievement Score (0–10)', fontweight='bold')
ax2.set_ylabel('Score')
add_value_labels(ax2, '{:.2f}')
style_ax(ax2)

#Avg casualty intensity ─────────────────────────────────────────────────
ax3.bar(gen_order, avg_intensity, color=bar_colors(gen_order), edgecolor='white')
ax3.set_title('Avg Casualty Intensity', fontweight='bold')
ax3.set_ylabel('Casualties / Total Troops')
add_value_labels(ax3, '{:.3f}')
style_ax(ax3)

#Underdog rate ──────────────────────────────────────────────────────────
ax4.bar(gen_order, underdog_rate, color=bar_colors(gen_order), edgecolor='white')
ax4.set_title('% Battles as Underdog (Force Ratio < 0.8)', fontweight='bold')
ax4.set_ylabel('%')
ax4.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=100, decimals=0, symbol='%'))
add_value_labels(ax4, '{:.0f}')
style_ax(ax4)

#Cluster distribution: Napoleon vs peers ────────────────────────────────
x = np.arange(8)
width = 0.35
bars1 = ax5.bar(x - width/2, nap_clusters, width, label='Napoleon',
                color=Napoleon_Color, alpha=0.85, edgecolor='white')
bars2 = ax5.bar(x + width/2, others_clusters, width, label='Other Generals',
                color=Other_Color, alpha=0.85, edgecolor='white')
ax5.set_xticks(x)
ax5.set_xticklabels([cluster_names[i] for i in range(8)], fontsize=6.5, rotation=35, ha='right')
ax5.set_title('Cluster Distribution: Napoleon vs Peers', fontweight='bold')
ax5.set_ylabel('% of Battles')
ax5.legend(fontsize=8)
style_ax(ax5)

#Napoleon win rate by cluster type 
colors_6 = [Napoleon_Color] * len(nap_win_by_cluster)
ax6.barh(nap_win_by_cluster.index, nap_win_by_cluster['mean'],
         color=colors_6, edgecolor='white', alpha=0.85)
for i, (val, cnt) in enumerate(zip(nap_win_by_cluster['mean'], nap_win_by_cluster['count'])):
    ax6.text(val + 1, i, f'{val:.0f}%  (n={cnt})', va='center', fontsize=8)
ax6.set_xlim(0, 115)
ax6.set_title("Napoleon's Win Rate by Cluster Type", fontweight='bold')
ax6.set_xlabel('Win Rate %')
ax6.set_facecolor('#F7F7F7')
ax6.spines[['top', 'right']].set_visible(False)

fig.savefig('./BattleML/data/napoleon_vs_generals.png', dpi=200, bbox_inches='tight')
plt.close()
print("Saved: napoleon_vs_generals.png")




summary = pd.DataFrame({
    'Battles':          gen_df.groupby('co_clean').size().reindex(gen_order),
    'Win Rate %':       win_rate.round(1),
    'Avg Ach':          avg_ach.round(2),
    'Avg Intensity':    avg_intensity.round(4),
    'Underdog %':       underdog_rate.round(1),
})
print("\n── General Comparison Summary ──")
print(summary.to_string())