import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from scipy.cluster.hierarchy import dendrogram, linkage

os.environ["LOKY_MAX_CPU_COUNT"] = "4"
warnings.filterwarnings("ignore")

# Setup visual style
sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["font.sans-serif"] = ["Segoe UI", "DejaVu Sans", "Arial"]

out_dir = os.path.join("reports", "figures")
os.makedirs(out_dir, exist_ok=True)

# 1. Load Data
df = pd.read_csv("data/customer_segmentation_data.csv")

# Standardize
features = ["Age", "Annual_Income_k", "Spending_Score", "Purchase_Frequency", "Recency_Days"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df[features])

# Optimal k=5
kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
df["Cluster"] = kmeans.fit_predict(X_scaled)

# Persona Mapping
cluster_stats = df.groupby("Cluster")[["Annual_Income_k", "Spending_Score"]].mean()
persona_dict = {}
for c_id, row in cluster_stats.iterrows():
    inc = row["Annual_Income_k"]
    score = row["Spending_Score"]
    if inc > 75 and score > 65:
        persona_dict[c_id] = "Elite VIP Champions"
    elif inc > 75 and score <= 45:
        persona_dict[c_id] = "Affluent Savers"
    elif inc <= 55 and score > 65:
        persona_dict[c_id] = "Young Trendsetters"
    elif inc <= 50 and score <= 45:
        persona_dict[c_id] = "Bargain Seekers"
    else:
        persona_dict[c_id] = "Core Balanced Families"

df["Segment"] = df["Cluster"].map(persona_dict)

palette = {
    "Elite VIP Champions": "#1E40AF",
    "Affluent Savers": "#0D9488",
    "Young Trendsetters": "#7C3AED",
    "Bargain Seekers": "#EA580C",
    "Core Balanced Families": "#059669"
}

# -------------------------------------------------------------
# Chart 1: Elbow Method & Silhouette Score
# -------------------------------------------------------------
k_range = range(2, 11)
wcss = []
sil_scores = []
for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    wcss.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, km.labels_))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5.5), dpi=300)

ax1.plot(k_range, wcss, marker='o', linewidth=2.5, color="#2563EB", markersize=7)
ax1.axvline(x=5, color="#DC2626", linestyle='--', linewidth=2, label="Optimal k = 5")
ax1.set_title("Elbow Method: Within-Cluster Sum of Squares (Inertia)", fontsize=12, fontweight="bold")
ax1.set_xlabel("Number of Clusters (k)", fontsize=11)
ax1.set_ylabel("Inertia (WCSS)", fontsize=11)
ax1.legend(frameon=True)
ax1.grid(True, linestyle="--", alpha=0.6)

ax2.plot(k_range, sil_scores, marker='s', linewidth=2.5, color="#10B981", markersize=7)
ax2.axvline(x=5, color="#DC2626", linestyle='--', linewidth=2, label="Optimal k = 5 (Score: 0.385)")
ax2.set_title("Silhouette Coefficient vs. Number of Clusters", fontsize=12, fontweight="bold")
ax2.set_xlabel("Number of Clusters (k)", fontsize=11)
ax2.set_ylabel("Average Silhouette Score", fontsize=11)
ax2.legend(frameon=True)
ax2.grid(True, linestyle="--", alpha=0.6)

plt.tight_layout()
fig1_path = os.path.join(out_dir, "01_elbow_silhouette_analysis.png")
plt.savefig(fig1_path, bbox_inches="tight")
plt.close()
print(f"Saved: {fig1_path}")

# -------------------------------------------------------------
# Chart 2: 2D Clusters: Income vs Spending Score
# -------------------------------------------------------------
plt.figure(figsize=(12, 7.5), dpi=300)
for seg_name, color in palette.items():
    sub = df[df["Segment"] == seg_name]
    plt.scatter(sub["Annual_Income_k"], sub["Spending_Score"], color=color, label=seg_name, s=65, alpha=0.82, edgecolors="none")

# Centroids
centroids = scaler.inverse_transform(kmeans.cluster_centers_)
inc_idx = features.index("Annual_Income_k")
score_idx = features.index("Spending_Score")
plt.scatter(centroids[:, inc_idx], centroids[:, score_idx], marker="X", s=280, color="black", edgecolor="#FBBF24", linewidth=2.5, label="Cluster Centroids", zorder=10)

plt.title("Customer Segments: Annual Income ($k) vs. Spending Score (1-100)", fontsize=14, fontweight="bold", pad=15)
plt.xlabel("Annual Income ($k)", fontsize=11)
plt.ylabel("Spending Score (1-100)", fontsize=11)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
fig2_path = os.path.join(out_dir, "02_clusters_income_vs_spending.png")
plt.savefig(fig2_path, bbox_inches="tight")
plt.close()
print(f"Saved: {fig2_path}")

# -------------------------------------------------------------
# Chart 3: PCA 2D Projection
# -------------------------------------------------------------
pca = PCA(n_components=2)
coords = pca.fit_transform(X_scaled)
df["PCA1"] = coords[:, 0]
df["PCA2"] = coords[:, 1]
var_exp = pca.explained_variance_ratio_

plt.figure(figsize=(11, 7), dpi=300)
for seg_name, color in palette.items():
    sub = df[df["Segment"] == seg_name]
    plt.scatter(sub["PCA1"], sub["PCA2"], color=color, label=seg_name, s=70, alpha=0.82)

plt.title(f"Principal Component Analysis (PCA) 2D Projection\n(Total Explained Variance: {sum(var_exp)*100:.1f}%)", fontsize=13, fontweight="bold", pad=15)
plt.xlabel(f"Principal Component 1 ({var_exp[0]*100:.1f}% variance)", fontsize=11)
plt.ylabel(f"Principal Component 2 ({var_exp[1]*100:.1f}% variance)", fontsize=11)
plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left", frameon=True, fontsize=10)
plt.grid(True, linestyle="--", alpha=0.6)
plt.tight_layout()
fig3_path = os.path.join(out_dir, "03_pca_2d_projection.png")
plt.savefig(fig3_path, bbox_inches="tight")
plt.close()
print(f"Saved: {fig3_path}")

# -------------------------------------------------------------
# Chart 4: Hierarchical Clustering Dendrogram
# -------------------------------------------------------------
sample_idx = np.random.RandomState(42).choice(len(X_scaled), size=120, replace=False)
linkage_mat = linkage(X_scaled[sample_idx], method='ward')

plt.figure(figsize=(13, 6), dpi=300)
dendrogram(linkage_mat, truncate_mode='lastp', p=20, leaf_rotation=90., leaf_font_size=9., show_contracted=True)
plt.axhline(y=12.0, color='#DC2626', linestyle='--', linewidth=2, label="Cluster Cut Threshold (k = 5)")
plt.title("Hierarchical Clustering Dendrogram (Ward Linkage)", fontsize=13, fontweight="bold", pad=15)
plt.xlabel("Sample Clusters / Index", fontsize=11)
plt.ylabel("Euclidean Distance", fontsize=11)
plt.legend(frameon=True)
plt.tight_layout()
fig4_path = os.path.join(out_dir, "04_hierarchical_dendrogram.png")
plt.savefig(fig4_path, bbox_inches="tight")
plt.close()
print(f"Saved: {fig4_path}")

# -------------------------------------------------------------
# Chart 5: Purchase Patterns & Customer Preferences Heatmap
# -------------------------------------------------------------
ct_category = pd.crosstab(df["Segment"], df["Preferred_Category"], normalize="index") * 100
ct_channel = pd.crosstab(df["Segment"], df["Preferred_Channel"], normalize="index") * 100

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5), dpi=300)

sns.heatmap(ct_category, annot=True, fmt=".1f", cmap="YlGnBu", cbar=True, ax=ax1, linewidths=0.8)
ax1.set_title("Product Category Preference by Segment (%)", fontsize=12, fontweight="bold", pad=12)
ax1.set_xlabel("Product Category", fontsize=10)
ax1.set_ylabel("Customer Segment", fontsize=10)

sns.heatmap(ct_channel, annot=True, fmt=".1f", cmap="BuPu", cbar=True, ax=ax2, linewidths=0.8)
ax2.set_title("Shopping Channel Preference by Segment (%)", fontsize=12, fontweight="bold", pad=12)
ax2.set_xlabel("Shopping Channel", fontsize=10)
ax2.set_ylabel("")

plt.tight_layout()
fig5_path = os.path.join(out_dir, "05_preferences_and_channels_heatmap.png")
plt.savefig(fig5_path, bbox_inches="tight")
plt.close()
print(f"Saved: {fig5_path}")

# -------------------------------------------------------------
# Chart 6: Segment Size vs. Revenue Contribution Comparison
# -------------------------------------------------------------
seg_rev = df.groupby("Segment").agg(
    Customers=("Customer_ID", "count"),
    Revenue=("Annual_Spend", "sum")
).reset_index()

seg_rev["Customer_Share"] = (seg_rev["Customers"] / len(df)) * 100
seg_rev["Revenue_Share"] = (seg_rev["Revenue"] / df["Annual_Spend"].sum()) * 100

fig, ax = plt.subplots(figsize=(12, 6), dpi=300)
x = np.arange(len(seg_rev))
width = 0.35

rects1 = ax.bar(x - width/2, seg_rev["Customer_Share"], width, label="Customer Base Share (%)", color="#3B82F6", edgecolor="none")
rects2 = ax.bar(x + width/2, seg_rev["Revenue_Share"], width, label="Revenue Contribution (%)", color="#10B981", edgecolor="none")

ax.set_title("Customer Share vs. Revenue Contribution by Segment (%)", fontsize=13, fontweight="bold", pad=15)
ax.set_xticks(x)
ax.set_xticklabels(seg_rev["Segment"], rotation=12, ha="right", fontsize=10)
ax.set_ylabel("Share (%)", fontsize=11)
ax.legend(frameon=True)
ax.grid(axis="y", linestyle="--", alpha=0.6)

for r in rects1:
    h = r.get_height()
    ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")
for r in rects2:
    h = r.get_height()
    ax.annotate(f"{h:.1f}%", xy=(r.get_x() + r.get_width()/2, h), xytext=(0, 3), textcoords="offset points", ha="center", va="bottom", fontsize=9, fontweight="bold")

plt.tight_layout()
fig6_path = os.path.join(out_dir, "06_customer_share_vs_revenue.png")
plt.savefig(fig6_path, bbox_inches="tight")
plt.close()
print(f"Saved: {fig6_path}")

# -------------------------------------------------------------
# Chart 7: Normalized Radar Chart of Segment Characteristics
# -------------------------------------------------------------
radar_vars = ["Age", "Annual_Income_k", "Spending_Score", "Purchase_Frequency", "Recency_Days"]
radar_means = df.groupby("Segment")[radar_vars].mean()
# Normalize 0 to 1
radar_norm = (radar_means - radar_means.min()) / (radar_means.max() - radar_means.min())

labels = ["Age", "Income ($k)", "Spending Score", "Frequency (Orders)", "Recency (Days)"]
num_vars = len(labels)
angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
angles += angles[:1]

fig, ax = plt.subplots(figsize=(8.5, 8.5), subplot_kw=dict(polar=True), dpi=300)
for seg_name, color in palette.items():
    vals = radar_norm.loc[seg_name].tolist()
    vals += vals[:1]
    ax.plot(angles, vals, color=color, linewidth=2, label=seg_name)
    ax.fill(angles, vals, color=color, alpha=0.12)

ax.set_theta_offset(np.pi / 2)
ax.set_theta_direction(-1)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(labels, fontsize=10, fontweight="bold")
ax.set_ylim(0, 1)
plt.title("Normalized Segment DNA Comparison (Radar Chart)", fontsize=13, fontweight="bold", pad=25)
plt.legend(bbox_to_anchor=(1.25, 1.1), loc="upper left", frameon=True, fontsize=9)
plt.tight_layout()
fig7_path = os.path.join(out_dir, "07_radar_segment_dna.png")
plt.savefig(fig7_path, bbox_inches="tight")
plt.close()
print(f"Saved: {fig7_path}")

print("🎉 All 7 high-resolution visualization charts successfully exported!")

