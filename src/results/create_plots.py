import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import sys
from pathlib import Path

# Get project root (two levels up from src/results/)
ROOT = Path(__file__).resolve().parents[2]

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Load data
top_10_pct_load_df = pd.read_csv(ROOT / "data" / "results" / "day_ahead_temp_uncertainty_filter_top_10_pct_load.csv")
test_rmse = pd.read_csv(ROOT / "data" / "results" / "day_ahead_temp_uncertainty_filter_none.csv")
season_df = pd.read_csv(ROOT / "data" / "results" / "day_ahead_temp_uncertainty_filter_season.csv")
hour_df = pd.read_csv(ROOT / "data" / "results" / "day_ahead_temp_uncertainty_filter_hour.csv")

# Plot settings

model_class_names = ["linears", "xgb_best_per_hour", "lin_xgb_best_per_hour"]
model_class_titles = {
    "linears": "Linear",
    "xgb_best_per_hour": "XGBoost",
    "lin_xgb_best_per_hour": "Hybrid"
}

title_font = 14
label_font = 14
tick_font = 12
legend_font = 12
line_width = 2.5
marker_size = 7
alpha_std = 0.2
temp_unc_limit = 5  # °F

palette = sns.color_palette("tab10", n_colors=len(model_class_names))

# Baseline RMSE Bar Plot
baseline_save_loc = ""

bar_colors = ['#1f77b4', '#ff7f0e']

total_rmse_vals = []
top10_rmse_vals = []

for model_name in model_class_names:

    # Baseline
    baseline_total_row = test_rmse[(test_rmse['model_name']==model_name) & (test_rmse['temp_uncertainty']==0)]
    baseline_total = baseline_total_row['test_error'].iloc[0] if not baseline_total_row.empty else np.nan
    total_rmse_vals.append(baseline_total)
    
    # Top 10%
    baseline_top10_row = top_10_pct_load_df[(top_10_pct_load_df['model_name']==model_name) & (top_10_pct_load_df['temp_uncertainty']==0)]
    baseline_top10 = baseline_top10_row['test_error'].iloc[0] if not baseline_top10_row.empty else np.nan
    top10_rmse_vals.append(baseline_top10)

x = np.arange(len(model_class_names))
width = 0.35

fig, ax = plt.subplots(figsize=(8,5))

bars_total = ax.bar(x - width/2, total_rmse_vals, width, label='All Hours', color=bar_colors[0])
bars_top10 = ax.bar(x + width/2, top10_rmse_vals, width, label='Top 10% Load', color=bar_colors[1])

ax.set_ylabel('RMSE (MW)', fontsize=label_font)
ax.set_xticks(x)
ax.set_xticklabels([model_class_titles[m] for m in model_class_names], fontsize=tick_font)
ax.legend(fontsize=legend_font)
ax.grid(axis='y', alpha=0.3)

for b in bars_total + bars_top10:
    height = b.get_height()
    ax.annotate(f'{height:.1f}',
                xy=(b.get_x() + b.get_width()/2, height),
                xytext=(0,3),
                textcoords="offset points",
                ha='center', va='bottom')

plt.suptitle("RMSE Comparison Under Ideal Conditions\n(All Hours vs Top 10% Load)", fontsize=title_font + 2)
plt.tight_layout()
fig.savefig(ROOT / "plots" / "baseline_top10_rmse_bar.png", dpi=300, bbox_inches="tight")


# Total vs Top 10% RMSE vs Temp Uncertainty

fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharex=True, sharey=True)

for i, (rmse_df, title) in enumerate([(test_rmse, "Total RMSE"), (top_10_pct_load_df, "Top 10% Load RMSE")]):
    ax = axes[i]
    for j, model_name in enumerate(model_class_names):
        df = rmse_df[rmse_df['model_name']==model_name].copy()
        df = df[df['temp_uncertainty']<=temp_unc_limit].sort_values('temp_uncertainty')
        ax.plot(df['temp_uncertainty'], df['test_error'],
                marker='o', linewidth=line_width, markersize=marker_size,
                color=palette[j], label=model_class_titles[model_name])
        if 'test_std' in df.columns:
            ax.fill_between(df['temp_uncertainty'],
                            df['test_error'] - df['test_std'],
                            df['test_error'] + df['test_std'],
                            alpha=alpha_std, color=palette[j])
    ax.set_title(f"{title} vs Temperature Uncertainty", fontsize=title_font)
    ax.set_xlabel("Temperature Uncertainty (°F)", fontsize=label_font)
    if i==0:
        ax.set_ylabel("RMSE (MW)", fontsize=label_font)
    ax.set_xlim(0, temp_unc_limit)
    ax.set_ylim(0, 400)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=tick_font)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.17, 0.715),
           title='Models', fontsize=legend_font, title_fontsize=legend_font)

plt.suptitle("Model Performance vs Temperature Forecast Uncertainty (2022 Test Set)", fontsize=title_font+2)
plt.tight_layout(rect=[0,0,0.88,1])
fig.savefig(ROOT / "plots" / "baseline_top10_rmse_uncertainty_plot.png", dpi=300, bbox_inches="tight")

# Isolate Top 10% Load RMSE plot

fig, ax = plt.subplots(figsize=(8,5))  # single plot, adjust size as needed

for j, model_name in enumerate(model_class_names):
    df = top_10_pct_load_df[top_10_pct_load_df['model_name']==model_name].copy()
    df = df[df['temp_uncertainty'] <= temp_unc_limit].sort_values('temp_uncertainty')
    
    ax.plot(df['temp_uncertainty'], df['test_error'],
            marker='o', linewidth=line_width, markersize=marker_size,
            color=palette[j], label=model_class_titles[model_name])
    
    if 'test_std' in df.columns:
        ax.fill_between(df['temp_uncertainty'],
                        df['test_error'] - df['test_std'],
                        df['test_error'] + df['test_std'],
                        alpha=alpha_std, color=palette[j])

ax.set_title("RMSE vs Temperature Uncertainty over Top 10% Load Hours", fontsize=title_font)
ax.set_xlabel("Temperature Uncertainty (°F)", fontsize=label_font)
ax.set_ylabel("RMSE (MW)", fontsize=label_font)
ax.set_xlim(0, temp_unc_limit)
ax.set_ylim(0, 400)
ax.grid(True, alpha=0.3)
ax.tick_params(axis='both', labelsize=tick_font)
ax.legend(fontsize=legend_font, loc="lower right")

plt.tight_layout()
fig.savefig(ROOT / "plots" / "top10_load_rmse_uncertainty_plot.png", dpi=300, bbox_inches="tight")

# Seasonal Absolute RMSE

fig, axes = plt.subplots(1, 2, figsize=(14,5), sharey=True)

for i, season_key in enumerate(['Winter','Summer']):
    ax = axes[i]
    df_slice = season_df[season_df['Season']==season_key]
    for j, model_name in enumerate(model_class_names):
        df = df_slice[df_slice['model_name']==model_name].copy()
        df = df[df['temp_uncertainty']<=temp_unc_limit].sort_values('temp_uncertainty')
        ax.plot(df['temp_uncertainty'], df['test_error'],
                marker='o', linewidth=line_width, markersize=marker_size,
                color=palette[j], label=model_class_titles[model_name])
        if 'test_std' in df.columns:
            ax.fill_between(df['temp_uncertainty'],
                            df['test_error'] - df['test_std'],
                            df['test_error'] + df['test_std'],
                            alpha=alpha_std, color=palette[j])
    ax.set_title(f"{season_key} Absolute RMSE", fontsize=title_font)
    ax.set_xlabel("Temperature Uncertainty (°F)", fontsize=label_font)
    ax.set_ylabel("RMSE (MW)", fontsize=label_font)
    ax.set_xlim(0, temp_unc_limit)
    ax.set_ylim(0, 500)
    ax.grid(True, alpha=0.3)
    ax.tick_params(axis='both', labelsize=tick_font)

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='center right', bbox_to_anchor=(0.17, 0.67),
           title='Models', fontsize=legend_font, title_fontsize=legend_font)


plt.suptitle("Model Performance vs Season (2022 Test Set)", fontsize=title_font+2)
plt.tight_layout(rect=[0,0,0.88,0.95])
fig.savefig(ROOT / "plots" / "seasonal_rmse_uncertainty_plot.png", dpi=300, bbox_inches="tight")


# Hourly RMSE Heatmap

hour_df_filtered = hour_df[hour_df['temp_uncertainty']<=temp_unc_limit]
vmin = hour_df_filtered['test_error'].min()
vmax = hour_df_filtered['test_error'].max()

fig, axes = plt.subplots(1, len(model_class_names), figsize=(18,6), sharey=True)
for i, model_name in enumerate(model_class_names):
    ax = axes[i]
    df = hour_df_filtered[hour_df_filtered['model_name']==model_name].copy()
    heatmap_data = df.pivot(index='Hour', columns='temp_uncertainty', values='test_error')
    sns.heatmap(heatmap_data, ax=ax, cmap='viridis', annot=False,
                vmin=vmin, vmax=vmax, cbar_kws={'label':'RMSE (MW)'})

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)  # Tick labels
    cbar.set_label('RMSE Increase (MW)', fontsize=14)  # Label

    ax.set_title(model_class_titles[model_name], fontsize = 17)
    ax.set_xlabel("Temperature Uncertainty (°F)", fontsize = 14)
    if i == 0:
        ax.set_ylabel("Hour of Day", fontsize = 14)
    else:
        ax.set_ylabel("")
plt.suptitle("Hourly Model Performance Across Temperature Uncertainty", fontsize=title_font + 8, y=1.05)
plt.tight_layout()

fig.savefig(ROOT / "plots" / "hourly_rmse_uncertainty_heatmap.png", dpi=300, bbox_inches="tight")



# Hourly RMSE Percentage Increase Heatmap
baseline = hour_df[hour_df['temp_uncertainty']==0][['model_name','Hour','test_error']].rename(columns={'test_error':'baseline_rmse'})

hour_rel_df = hour_df.merge(baseline, on=['model_name','Hour'], how='left')

# Compute percentage increase
hour_rel_df['rmse_pct_increase'] = 100 * (hour_rel_df['test_error'] - hour_rel_df['baseline_rmse']) / hour_rel_df['baseline_rmse']

hour_rel_df_filtered = hour_rel_df[hour_rel_df['temp_uncertainty']<=temp_unc_limit]

vmin = hour_rel_df_filtered['rmse_pct_increase'].min()
vmax = hour_rel_df_filtered['rmse_pct_increase'].max()

fig, axes = plt.subplots(1, len(model_class_names), figsize=(18,6), sharey=True)
for i, model_name in enumerate(model_class_names):
    ax = axes[i]
    df = hour_rel_df_filtered[hour_rel_df_filtered['model_name']==model_name].copy()
    heatmap_data = df.pivot(index='Hour', columns='temp_uncertainty', values='rmse_pct_increase')
    
    sns.heatmap(heatmap_data, ax=ax, cmap='Reds', annot=False,
                vmin=vmin, vmax=vmax, cbar_kws={'label':'RMSE Increase (%)'})

    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(labelsize=14)  # Tick labels
    cbar.set_label('RMSE Increase (%)', fontsize=14)  # Label

    ax.set_title(model_class_titles[model_name], fontsize=17)
    ax.set_xlabel("Temperature Uncertainty (°F)", fontsize=14)
    if i == 0:
        ax.set_ylabel("Hour of Day", fontsize=14)
    else:
        ax.set_ylabel("")

plt.suptitle("Hourly Fragility to Temperature Uncertainty (Percentage Increase from Baseline)", fontsize=title_font + 8, y=1.05)
plt.tight_layout()
fig.savefig(ROOT / "plots" / "hourly_fragility_pct_rmse_uncertainty_heatmap.png", dpi=300, bbox_inches="tight")