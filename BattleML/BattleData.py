import pandas as pd
import numpy as np

Load_Path = './BattleML/CDB90/data'

battles         = pd.read_csv(f"{Load_Path}/battles.csv")
belligerents    = pd.read_csv(f"{Load_Path}/belligerents.csv")
durations       = pd.read_csv(f"{Load_Path}/battle_durations.csv")
front_widths    = pd.read_csv(f"{Load_Path}/front_widths.csv")
terrain         = pd.read_csv(f"{Load_Path}/terrain.csv")
weather         = pd.read_csv(f"{Load_Path}/weather.csv")
battle_actors   = pd.read_csv(f"{Load_Path}/battle_actors.csv")

# ── Pivot belligerents into attacker / defender ──────────────────────────────
att = belligerents[belligerents['attacker'] == 1].add_prefix('att_').rename(columns={'att_isqno': 'isqno'})
dfd = belligerents[belligerents['attacker'] == 0].add_prefix('def_').rename(columns={'def_isqno': 'isqno'})

# ── Flat join ────────────────────────────────────────────────────────────────
df = (battles
      .merge(att, on='isqno')
      .merge(dfd, on='isqno')
      .merge(durations[['isqno', 'duration1']], on='isqno', how='left')
      .merge(front_widths.groupby('isqno')[['wofa', 'wofd']].first().reset_index(), on='isqno', how='left')
      .merge(terrain.groupby('isqno').first().reset_index(), on='isqno', how='left')
      .merge(weather.groupby('isqno').first().reset_index(), on='isqno', how='left')
)

# ── Select features ──────────────────────────────────────────────────────────
feature_cols = [
    'isqno', 'name', 'war', 'war4',
    'att_str', 'def_str', 'att_cas', 'def_cas',
    'att_cav', 'def_cav', 'att_arty', 'def_arty',
    'att_tank', 'def_tank',
    'att_ach', 'def_ach',
    'att_pri1', 'def_pri1',
    'duration1', 'wofa', 'wofd',
    'terra1', 'wx1',
    'surpa', 'morala', 'momnta', 'techa', 'inita', 'mobila',
]

df_feat = df[feature_cols].copy()
df_feat[['att_tank', 'def_tank']] = df_feat[['att_tank', 'def_tank']].fillna(0)

# ── Impute ───────────────────────────────────────────────────────────────────
impute_cols = [
    'att_arty', 'def_arty', 'att_cav', 'def_cav',
    'inita', 'mobila', 'morala', 'techa', 'momnta',
    'wofa', 'wofd', 'terra1', 'surpa',
    'att_cas', 'def_cas', 'att_str', 'def_str',
    'duration1', 'wx1', 'att_pri1', 'def_pri1',
]

for col in impute_cols:
    if df_feat[col].dtype == object or pd.api.types.is_string_dtype(df_feat[col]):
        df_feat[col] = df_feat[col].fillna(df_feat[col].mode()[0])
    else:
        df_feat[col] = df_feat[col].fillna(df_feat[col].median())

print(f"Nulls remaining: {df_feat.isnull().sum().sum()}")
print(f"Shape: {df_feat.shape}")

# ── Engineer features ────────────────────────────────────────────────────────
df_feat['force_ratio']        = df_feat['att_str'] / df_feat['def_str']
df_feat['att_loss_pct']       = df_feat['att_cas'] / df_feat['att_str']
df_feat['def_loss_pct']       = df_feat['def_cas'] / df_feat['def_str']
df_feat['exchange_ratio']     = df_feat['att_cas'] / df_feat['def_cas'].replace(0, np.nan)
df_feat['total_troops']       = df_feat['att_str'] + df_feat['def_str']
df_feat['casualty_intensity'] = (df_feat['att_cas'] + df_feat['def_cas']) / df_feat['total_troops']
df_feat['attacker_underdog']  = (df_feat['force_ratio'] < 0.80).astype(int)
df_feat['ach_diff']           = df_feat['att_ach'] - df_feat['def_ach']

# ── Cap outliers at 99th percentile ─────────────────────────────────────────
for col in ['exchange_ratio', 'att_loss_pct', 'def_loss_pct', 'casualty_intensity', 'force_ratio']:
    df_feat[col] = df_feat[col].clip(upper=df_feat[col].quantile(0.99))

# ── Log transform skewed columns ─────────────────────────────────────────────
for col in ['att_str', 'def_str', 'att_cas', 'def_cas', 'total_troops', 'exchange_ratio', 'force_ratio', 'duration1']:
    df_feat[f'log_{col}'] = np.log1p(df_feat[col])

print(df_feat[['force_ratio', 'att_loss_pct', 'def_loss_pct', 'exchange_ratio', 'casualty_intensity', 'ach_diff']].describe())

import os
os.makedirs('./BattleML/data', exist_ok=True)

df_feat.to_csv('./BattleML/data/wars.csv', index=False)
print("Saved: wars.csv")