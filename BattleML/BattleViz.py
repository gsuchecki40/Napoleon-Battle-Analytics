import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

df = pd.read_csv('./BattleML/data/battles_clustered.csv')

cluster_names = {
    0: "The Grind",
    1: "Defender's Fortress",
    2: "Lightning Strike",
    3: "Skirmish",
    4: "Annihilation",
    5: "Grand Carnage",
    6: "Repulse",
    7: "Industrial Slaughter",
}
df['cluster_label'] = df['kmeans'].map(cluster_names)

palette = {
    0: "#4878CF",
    1: "#D65F5F",
    2: "#6ACC65",
    3: "#B47CC7",
    4: "#C4AD66",
    5: "#77BEDB",
    6: "#F17F29",
    7: "#1B1B1B",
}

napoleonic_wars = [
    "War of the First Coalition of 1792-1797",
    "War of the Second Coalition of 1798-1802",
    "War of the Third Coalition of 1805",
    "War of the Fourth Coalition of 1806-1807",
    "War of the Fifth Coalition of 1809",
    "French Invasion of Russia of 1812",
    "War of the Sixth Coalition of 1812-1814",
    "Peninsular War of 1808-1814",
    "Hundred Days of 1814",
]
df['is_napoleonic'] = df['war4'].isin(napoleonic_wars)

# Graphs

fig, ax = plt.subplots(figsize=(16, 11))

for cluster_id, name in cluster_names.items():
    mask = df['kmeans'] == cluster_id
    ax.scatter(
        df.loc[mask, 'umap_x'], df.loc[mask, 'umap_y'],
        c=palette[cluster_id], label=f"{cluster_id}: {name}",
        s=45, alpha=0.75, edgecolors='white', linewidths=0.3, zorder=2
    )

for _, row in df.iterrows():
    ax.annotate(
        row['name'],
        (row['umap_x'], row['umap_y']),
        fontsize=3.5, alpha=0.55, zorder=3,
        xytext=(2, 2), textcoords='offset points'
    )

ax.legend(loc='upper left', fontsize=8, framealpha=0.9,
          title='Cluster', title_fontsize=9)
ax.set_title('Historical Battle Clusters (UMAP)', fontsize=15, fontweight='bold', pad=12)
ax.set_xlabel('UMAP Dimension 1', fontsize=10)
ax.set_ylabel('UMAP Dimension 2', fontsize=10)
ax.set_facecolor('#F7F7F7')
fig.tight_layout()
fig.savefig('./BattleML/data/viz_umap_clusters.png', dpi=200)
plt.close()
print("Saved: viz_umap_clusters.png")

# Napoleonic Wars Highlight

fig, ax = plt.subplots(figsize=(16, 11))

mask_bg = ~df['is_napoleonic']
ax.scatter(
    df.loc[mask_bg, 'umap_x'], df.loc[mask_bg, 'umap_y'],
    c='#CCCCCC', s=30, alpha=0.35, edgecolors='none', zorder=1, label='Other battles'
)

mask_nap = df['is_napoleonic']
for cluster_id, name in cluster_names.items():
    mask = mask_nap & (df['kmeans'] == cluster_id)
    if mask.sum() == 0:
        continue
    ax.scatter(
        df.loc[mask, 'umap_x'], df.loc[mask, 'umap_y'],
        c=palette[cluster_id], s=80, alpha=0.95,
        edgecolors='black', linewidths=0.5, zorder=3,
        label=f"{cluster_id}: {name}"
    )

for _, row in df[mask_nap].iterrows():
    ax.annotate(
        row['name'],
        (row['umap_x'], row['umap_y']),
        fontsize=5.5, alpha=0.9, fontweight='bold', zorder=4,
        xytext=(3, 3), textcoords='offset points'
    )

ax.legend(loc='upper left', fontsize=8, framealpha=0.9,
          title='Cluster', title_fontsize=9)
ax.set_title('Napoleonic Battles Highlighted by Cluster (UMAP)',
             fontsize=15, fontweight='bold', pad=12)
ax.set_xlabel('UMAP Dimension 1', fontsize=10)
ax.set_ylabel('UMAP Dimension 2', fontsize=10)
ax.set_facecolor('#F7F7F7')
fig.tight_layout()
fig.savefig('./BattleML/data/viz_umap_napoleonic.png', dpi=200)
plt.close()
print("Saved: viz_umap_napoleonic.png")
