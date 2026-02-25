from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
import umap
import hdbscan
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('./BattleML/data/wars.csv')

features = [
    'log_att_str', 'log_def_str', 'log_att_cas', 'log_def_cas',
    'log_total_troops', 'log_force_ratio', 'log_exchange_ratio',
    'att_loss_pct', 'def_loss_pct', 'casualty_intensity',
    'ach_diff', 'attacker_underdog',
    'log_duration1', 'wofa', 'wofd',
    'surpa', 'morala', 'momnta', 'techa', 'inita', 'mobila',
]

X = df[features].astype(float)
X = SimpleImputer(strategy='median').fit_transform(X)
X = StandardScaler().fit_transform(X)

pca = PCA(n_components=min(10, X.shape[1]), random_state=42).fit_transform(X)
emb = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=42).fit_transform(pca)
kmeans = KMeans(n_clusters=8, random_state=42).fit_predict(pca)
hdb = hdbscan.HDBSCAN(min_cluster_size=5).fit_predict(pca)

df['umap_x'], df['umap_y'] = emb[:, 0], emb[:, 1]
df['kmeans'], df['hdbscan'] = kmeans, hdb

df.to_csv('./BattleML/data/battles_clustered.csv', index=False)

plt.figure(figsize=(12, 6))
sns.scatterplot(data=df, x='umap_x', y='umap_y', hue='kmeans', palette='tab10', s=60)
for _, row in df.iterrows():
    plt.annotate(row['name'], (row['umap_x'], row['umap_y']), fontsize=4, alpha=0.5)
plt.savefig('./BattleML/data/battleclusters_umap.png', dpi=200)
plt.close()

print(df['kmeans'].value_counts().sort_index())

for cluster in sorted(df['kmeans'].unique()):
    print(f"\n{'='*50}")
    print(f"CLUSTER {cluster} ({len(df[df['kmeans']==cluster])} battles)")
    print(df[df['kmeans']==cluster][['name','war4']].head(8).to_string())

print(df.groupby('kmeans')[['log_att_str', 'log_def_str', 'casualty_intensity', 
                             'ach_diff', 'duration1']].median().round(3))