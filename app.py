import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import silhouette_score, davies_bouldin_score
from sklearn.decomposition import PCA
import sqlite3
import os

# ---------------------------------------------------------
# Page Configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="Customer Segmentation & Behavioral Analytics",
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom Modern CSS Styling
# ---------------------------------------------------------
st.markdown("""
<style>
    /* Metric Card Styling */
    .metric-card {
        background: linear-gradient(135deg, #ffffff 0%, #f8fafc 100%);
        border: 1px solid #e2e8f0;
        border-radius: 12px;
        padding: 18px 20px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.08);
    }
    .metric-title {
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.06em;
        color: #64748b;
        margin-bottom: 4px;
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #0f172a;
        margin-bottom: 2px;
    }
    .metric-sub {
        font-size: 0.8rem;
        font-weight: 600;
    }
    .badge-positive {
        color: #16a34a;
        background-color: #dcfce7;
        padding: 2px 8px;
        border-radius: 9999px;
    }
    .badge-neutral {
        color: #2563eb;
        background-color: #dbeafe;
        padding: 2px 8px;
        border-radius: 9999px;
    }
    .badge-accent {
        color: #7c3aed;
        background-color: #ede9fe;
        padding: 2px 8px;
        border-radius: 9999px;
    }
    .persona-card {
        border-radius: 12px;
        padding: 20px;
        margin-bottom: 18px;
        border-left: 6px solid #2563eb;
        background: #f8fafc;
        box-shadow: 0 2px 4px rgba(0,0,0,0.04);
    }
    .main-title {
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #0f172a;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Data Ingestion Helper
# ---------------------------------------------------------
@st.cache_data
def load_default_data():
    csv_path = "data/customer_segmentation_data.csv"
    if os.path.exists(csv_path):
        return pd.read_csv(csv_path)
    db_path = "data/customer_segmentation.db"
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("SELECT * FROM customers", conn)
        conn.close()
        return df
    raise FileNotFoundError("Customer dataset missing. Run generate_data.py first.")

# ---------------------------------------------------------
# Sidebar: Controls & Data Source
# ---------------------------------------------------------
st.sidebar.markdown("## 🎯 Control Center")
data_source_opt = st.sidebar.radio("Data Ingestion Source", ["Benchmark Dataset", "SQLite Database", "Upload Custom File"])

df_raw = None
if data_source_opt == "Benchmark Dataset":
    df_raw = load_default_data()
elif data_source_opt == "SQLite Database":
    db_path = "data/customer_segmentation.db"
    if os.path.exists(db_path):
        conn = sqlite3.connect(db_path)
        df_raw = pd.read_sql_query("SELECT * FROM customers", conn)
        conn.close()
    else:
        st.sidebar.error("SQLite database file not found.")
        df_raw = load_default_data()
else:
    uploaded_file = st.sidebar.file_uploader("Upload CSV or Excel", type=["csv", "xlsx"])
    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith(".csv"):
                df_raw = pd.read_csv(uploaded_file)
            else:
                df_raw = pd.read_excel(uploaded_file)
            st.sidebar.success(f"Loaded {len(df_raw)} records!")
        except Exception as e:
            st.sidebar.error(f"Error loading file: {e}")
            df_raw = load_default_data()
    else:
        st.sidebar.info("Using default benchmark dataset until file is uploaded.")
        df_raw = load_default_data()

# ---------------------------------------------------------
# Sidebar Slicers & Filters
# ---------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 🔍 Global Slicers")

all_regions = ["All"] + sorted(list(df_raw["Region"].dropna().unique())) if "Region" in df_raw.columns else ["All"]
sel_region = st.sidebar.selectbox("Geographic Region", all_regions)

all_channels = ["All"] + sorted(list(df_raw["Preferred_Channel"].dropna().unique())) if "Preferred_Channel" in df_raw.columns else ["All"]
sel_channel = st.sidebar.selectbox("Preferred Channel", all_channels)

all_genders = ["All"] + sorted(list(df_raw["Gender"].dropna().unique())) if "Gender" in df_raw.columns else ["All"]
sel_gender = st.sidebar.selectbox("Gender Demographic", all_genders)

# Filter dataset
df_filtered = df_raw.copy()
if sel_region != "All":
    df_filtered = df_filtered[df_filtered["Region"] == sel_region]
if sel_channel != "All":
    df_filtered = df_filtered[df_filtered["Preferred_Channel"] == sel_channel]
if sel_gender != "All":
    df_filtered = df_filtered[df_filtered["Gender"] == sel_gender]

if len(df_filtered) == 0:
    st.warning("No customers match the current filter criteria. Resetting filters.")
    df_filtered = df_raw.copy()

# ---------------------------------------------------------
# Machine Learning Clustering Pipeline
# ---------------------------------------------------------
clustering_cols = ["Age", "Annual_Income_k", "Spending_Score", "Purchase_Frequency", "Recency_Days"]
avail_cols = [c for c in clustering_cols if c in df_filtered.columns]

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_filtered[avail_cols])

# Dynamic hyperparameter in sidebar
st.sidebar.markdown("---")
st.sidebar.markdown("### 🧪 Clustering Hyperparameters")
num_clusters = st.sidebar.slider("Number of Clusters (k)", min_value=2, max_value=8, value=5, step=1)
algorithm_choice = st.sidebar.selectbox("Clustering Algorithm", ["K-Means", "Agglomerative Hierarchical"])

if algorithm_choice == "K-Means":
    model = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_labels = model.fit_predict(X_scaled)
else:
    model = AgglomerativeClustering(n_clusters=num_clusters)
    cluster_labels = model.fit_predict(X_scaled)

df_filtered["Cluster"] = cluster_labels

# Compute Metrics
sil_score = silhouette_score(X_scaled, cluster_labels) if len(np.unique(cluster_labels)) > 1 else 0
db_index = davies_bouldin_score(X_scaled, cluster_labels) if len(np.unique(cluster_labels)) > 1 else 0

# Assign interpretable business personas
cluster_profile_means = df_filtered.groupby("Cluster")[["Annual_Income_k", "Spending_Score"]].mean()
persona_map = {}
for c_idx, row in cluster_profile_means.iterrows():
    inc = row["Annual_Income_k"]
    score = row["Spending_Score"]
    if inc > 75 and score > 65:
        persona_map[c_idx] = "Elite VIP Champions"
    elif inc > 75 and score <= 45:
        persona_map[c_idx] = "Affluent Savers"
    elif inc <= 55 and score > 65:
        persona_map[c_idx] = "Young Trendsetters"
    elif inc <= 50 and score <= 45:
        persona_map[c_idx] = "Bargain Seekers"
    else:
        persona_map[c_idx] = f"Segment {c_idx+1} (Balanced)"

df_filtered["Cluster_Persona"] = df_filtered["Cluster"].map(persona_map)

# RFM Scoring
df_filtered["R_Score"] = pd.qcut(df_filtered["Recency_Days"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
df_filtered["F_Score"] = pd.qcut(df_filtered["Purchase_Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
df_filtered["M_Score"] = pd.qcut(df_filtered["Annual_Spend"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
df_filtered["RFM_Composite"] = df_filtered["R_Score"].astype(str) + df_filtered["F_Score"].astype(str) + df_filtered["M_Score"].astype(str)

def get_rfm_segment(row):
    r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
    if r >= 4 and f >= 4 and m >= 4:
        return "Champions"
    elif f >= 4 and m >= 3:
        return "Loyal Customers"
    elif r >= 4 and f in [2, 3]:
        return "Potential Loyalists"
    elif r >= 4 and f <= 2:
        return "New Customers"
    elif r in [2, 3] and f in [2, 3]:
        return "Need Attention"
    elif r <= 2 and f >= 3:
        return "At Risk"
    elif r == 1 and f >= 4:
        return "Can't Lose Them"
    else:
        return "Hibernating / Churned"

df_filtered["RFM_Segment"] = df_filtered.apply(get_rfm_segment, axis=1)

# PCA Projection for 2D & 3D visualizations
pca_2d = PCA(n_components=2)
coords_2d = pca_2d.fit_transform(X_scaled)
df_filtered["PCA1"] = coords_2d[:, 0]
df_filtered["PCA2"] = coords_2d[:, 1]
pca_var_exp = pca_2d.explained_variance_ratio_

pca_3d = PCA(n_components=3)
coords_3d = pca_3d.fit_transform(X_scaled)
df_filtered["PCA_3D_1"] = coords_3d[:, 0]
df_filtered["PCA_3D_2"] = coords_3d[:, 1]
df_filtered["PCA_3D_3"] = coords_3d[:, 2]

# ---------------------------------------------------------
# Main App Header
# ---------------------------------------------------------
st.markdown("<h1 class='main-title'>🎯 Customer Segmentation & Behavioral Analytics</h1>", unsafe_allow_html=True)
st.markdown(
    "**End-to-End Analytics Platform:** Machine Learning Clustering | RFM Behavioral Scoring | Segment Personas & Strategic Playbook"
)
st.markdown("---")

# ---------------------------------------------------------
# Executive KPI Cards (Row 1)
# ---------------------------------------------------------
col1, col2, col3, col4, col5, col6 = st.columns(6)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Active Customers</div>
        <div class="metric-value">{len(df_filtered):,}</div>
        <div class="metric-sub"><span class="badge-neutral">Filtered Base</span></div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    tot_rev = df_filtered["Annual_Spend"].sum()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Annual Spend</div>
        <div class="metric-value">${tot_rev/1e6:.2f}M</div>
        <div class="metric-sub"><span class="badge-positive">${tot_rev/len(df_filtered):,.0f} / cust</span></div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    avg_inc = df_filtered["Annual_Income_k"].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Avg Annual Income</div>
        <div class="metric-value">${avg_inc:.1f}k</div>
        <div class="metric-sub"><span class="badge-neutral">Household Median</span></div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_score = df_filtered["Spending_Score"].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Avg Spending Score</div>
        <div class="metric-value">{avg_score:.1f} <span style="font-size:1rem;color:#64748b;">/ 100</span></div>
        <div class="metric-sub"><span class="badge-accent">Index Scale</span></div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    avg_rec = df_filtered["Recency_Days"].mean()
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Average Recency</div>
        <div class="metric-value">{avg_rec:.1f}d</div>
        <div class="metric-sub"><span class="badge-neutral">Days Since Order</span></div>
    </div>
    """, unsafe_allow_html=True)

with col6:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Silhouette Score</div>
        <div class="metric-value">{sil_score:.3f}</div>
        <div class="metric-sub"><span class="badge-positive">Cluster Quality</span></div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ---------------------------------------------------------
# Navigation Tabs
# ---------------------------------------------------------
tabs = st.tabs([
    "📊 Executive Overview",
    "📈 Exploratory Data Analysis",
    "🧪 ML Clustering Lab",
    "🎯 RFM CRM Segmentation",
    "👤 Customer Personas & Playbook",
    "🔮 Predict New Customer Segment"
])

# ---------------------------------------------------------
# Tab 1: Executive Overview & Raw Data
# ---------------------------------------------------------
with tabs[0]:
    st.markdown("### 📌 Executive Segment Summary")
    st.markdown("High-level breakdown of customer volumes, aggregate revenue, and behavioral characteristics across clusters.")
    
    col_t1a, col_t1b = st.columns([3, 2])
    
    with col_t1a:
        summary_df = df_filtered.groupby("Cluster_Persona").agg(
            Customers=("Customer_ID", "count"),
            Total_Revenue=("Annual_Spend", "sum"),
            Avg_Income=("Annual_Income_k", "mean"),
            Avg_Spend_Score=("Spending_Score", "mean"),
            Avg_Frequency=("Purchase_Frequency", "mean"),
            Avg_Recency=("Recency_Days", "mean")
        ).reset_index()
        summary_df["Customer_Share_%"] = (summary_df["Customers"] / len(df_filtered)) * 100
        summary_df["Revenue_Share_%"] = (summary_df["Total_Revenue"] / tot_rev) * 100
        
        fig_bar = px.bar(
            summary_df,
            x="Cluster_Persona",
            y=["Customer_Share_%", "Revenue_Share_%"],
            barmode="group",
            title="Customer Base Share vs. Revenue Contribution Share (%)",
            labels={"value": "Percentage (%)", "Cluster_Persona": "Segment", "variable": "Metric"},
            color_discrete_sequence=["#3b82f6", "#10b981"]
        )
        fig_bar.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=360)
        st.plotly_chart(fig_bar, use_container_width=True)

    with col_t1b:
        fig_pie = px.pie(
            summary_df,
            names="Cluster_Persona",
            values="Total_Revenue",
            title="Total Revenue Contribution by Segment ($)",
            hole=0.45,
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_pie.update_layout(margin=dict(l=20, r=20, t=40, b=20), height=360)
        st.plotly_chart(fig_pie, use_container_width=True)

    st.markdown("#### 📋 Segment Metric Benchmark Table")
    formatted_summary = summary_df.copy()
    formatted_summary["Total_Revenue"] = formatted_summary["Total_Revenue"].apply(lambda x: f"${x:,.0f}")
    formatted_summary["Avg_Income"] = formatted_summary["Avg_Income"].apply(lambda x: f"${x:.1f}k")
    formatted_summary["Avg_Spend_Score"] = formatted_summary["Avg_Spend_Score"].apply(lambda x: f"{x:.1f}")
    formatted_summary["Avg_Frequency"] = formatted_summary["Avg_Frequency"].apply(lambda x: f"{x:.1f} orders")
    formatted_summary["Avg_Recency"] = formatted_summary["Avg_Recency"].apply(lambda x: f"{x:.1f} days")
    formatted_summary["Customer_Share_%"] = formatted_summary["Customer_Share_%"].apply(lambda x: f"{x:.1f}%")
    formatted_summary["Revenue_Share_%"] = formatted_summary["Revenue_Share_%"].apply(lambda x: f"{x:.1f}%")
    st.dataframe(formatted_summary, use_container_width=True, hide_index=True)

    with st.expander("📂 View Audited Customer Records (Search & Filter)"):
        search_kw = st.text_input("Search Customer by Name, ID, or City", "")
        view_df = df_filtered.copy()
        if search_kw:
            view_df = view_df[
                view_df["Customer_Name"].str.contains(search_kw, case=False, na=False) |
                view_df["Customer_ID"].str.contains(search_kw, case=False, na=False) |
                view_df["City"].str.contains(search_kw, case=False, na=False)
            ]
        st.dataframe(view_df.head(200), use_container_width=True)
        csv_download = view_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Filtered Customer Dataset (CSV)",
            data=csv_download,
            file_name="segmented_customers_export.csv",
            mime="text/csv"
        )

# ---------------------------------------------------------
# Tab 2: Exploratory Data Analysis (EDA)
# ---------------------------------------------------------
with tabs[1]:
    st.markdown("### 📈 Comprehensive Exploratory Data Analysis")
    st.markdown("Examine demographic distributions, purchasing frequency patterns, and feature correlations.")

    col_e1, col_e2 = st.columns(2)
    with col_e1:
        feature_to_plot = st.selectbox(
            "Select Numerical Metric to Inspect Distribution",
            ["Annual_Income_k", "Spending_Score", "Age", "Annual_Spend", "Purchase_Frequency", "Recency_Days"]
        )
        fig_dist = px.histogram(
            df_filtered,
            x=feature_to_plot,
            color="Cluster_Persona",
            marginal="box",
            nbins=30,
            title=f"Distribution of {feature_to_plot} across Segments",
            color_discrete_sequence=px.colors.qualitative.Set2
        )
        fig_dist.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_dist, use_container_width=True)

    with col_e2:
        cat_to_plot = st.selectbox(
            "Select Categorical Attribute to Inspect Volume",
            ["Preferred_Category", "Preferred_Channel", "Payment_Method", "Discount_Sensitivity", "Region"]
        )
        fig_cat = px.histogram(
            df_filtered,
            x=cat_to_plot,
            color="Cluster_Persona",
            barmode="stack",
            title=f"Customer Volume by {cat_to_plot}",
            color_discrete_sequence=px.colors.qualitative.Safe
        )
        fig_cat.update_layout(height=420, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_cat, use_container_width=True)

    st.markdown("#### 🔗 Correlation Matrix Heatmap")
    num_fields = ["Age", "Annual_Income_k", "Spending_Score", "Annual_Spend", "Purchase_Frequency", "Recency_Days", "Average_Order_Value"]
    corr_df = df_filtered[num_fields].corr()
    fig_corr = px.imshow(
        corr_df,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Pairwise Pearson Correlation Heatmap"
    )
    fig_corr.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
    st.plotly_chart(fig_corr, use_container_width=True)

# ---------------------------------------------------------
# Tab 3: ML Clustering Lab
# ---------------------------------------------------------
with tabs[2]:
    st.markdown("### 🧪 Machine Learning Clustering Lab")
    st.markdown(
        "Evaluate mathematical clustering quality, explore optimal cluster metrics (Elbow & Silhouette), and visualize 2D/3D customer boundaries."
    )

    # Elbow and Silhouette evaluation curve
    with st.expander("📐 View Optimal Cluster Evaluation Curve (Elbow Method & Silhouette Scores)", expanded=True):
        col_c1, col_c2 = st.columns(2)
        k_vals = list(range(2, 9))
        wcss_vals = []
        sil_vals = []
        
        for k_test in k_vals:
            km_temp = KMeans(n_clusters=k_test, random_state=42, n_init=5)
            km_temp.fit(X_scaled)
            wcss_vals.append(km_temp.inertia_)
            sil_vals.append(silhouette_score(X_scaled, km_temp.labels_))

        with col_c1:
            fig_elbow = go.Figure()
            fig_elbow.add_trace(go.Scatter(x=k_vals, y=wcss_vals, mode="lines+markers", marker=dict(size=8, color="#2563eb"), line=dict(width=2.5)))
            fig_elbow.add_vline(x=num_clusters, line_dash="dash", line_color="#ef4444", annotation_text=f"Current k={num_clusters}")
            fig_elbow.update_layout(title="Elbow Curve (Within-Cluster Sum of Squares / Inertia)", xaxis_title="Clusters (k)", yaxis_title="WCSS", height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_elbow, use_container_width=True)

        with col_c2:
            fig_sil = go.Figure()
            fig_sil.add_trace(go.Scatter(x=k_vals, y=sil_vals, mode="lines+markers", marker=dict(size=8, color="#10b981"), line=dict(width=2.5)))
            fig_sil.add_vline(x=num_clusters, line_dash="dash", line_color="#ef4444", annotation_text=f"Current k={num_clusters}")
            fig_sil.update_layout(title="Silhouette Coefficient vs. k (Peak indicates optimal cohesion)", xaxis_title="Clusters (k)", yaxis_title="Silhouette Score", height=320, margin=dict(l=20, r=20, t=40, b=20))
            st.plotly_chart(fig_sil, use_container_width=True)

    st.markdown("---")

    # 2D and 3D Cluster Visualizations
    col_v1, col_v2 = st.columns(2)

    with col_v1:
        st.markdown("#### 🗺️ 2D Income vs. Spending Score Clustering")
        fig_2d = px.scatter(
            df_filtered,
            x="Annual_Income_k",
            y="Spending_Score",
            color="Cluster_Persona",
            hover_data=["Customer_Name", "Age", "Annual_Spend", "Purchase_Frequency", "Recency_Days"],
            title="Annual Income ($k) vs. Spending Score (1-100)",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        # Add centroids if K-Means
        if algorithm_choice == "K-Means":
            centroids = scaler.inverse_transform(model.cluster_centers_)
            inc_i = avail_cols.index("Annual_Income_k")
            score_i = avail_cols.index("Spending_Score")
            fig_2d.add_trace(go.Scatter(
                x=centroids[:, inc_i],
                y=centroids[:, score_i],
                mode="markers",
                marker=dict(symbol="x", size=14, color="black", line=dict(width=2, color="yellow")),
                name="Cluster Centroids"
            ))
        fig_2d.update_layout(height=450, margin=dict(l=20, r=20, t=40, b=20))
        st.plotly_chart(fig_2d, use_container_width=True)

    with col_v2:
        st.markdown("#### 🌐 3D Interactive Multi-Dimensional Space")
        fig_3d = px.scatter_3d(
            df_filtered,
            x="PCA_3D_1",
            y="PCA_3D_2",
            z="PCA_3D_3",
            color="Cluster_Persona",
            hover_data=["Customer_Name", "Annual_Income_k", "Spending_Score", "Annual_Spend"],
            title="3D PCA Dimensionality Reduction Projection",
            color_discrete_sequence=px.colors.qualitative.Bold
        )
        fig_3d.update_layout(height=450, margin=dict(l=10, r=10, t=30, b=10))
        st.plotly_chart(fig_3d, use_container_width=True)

    # Spider / Radar Chart
    st.markdown("#### 🕸️ Cluster Characteristic Radar Profiles")
    radar_cols = ["Age", "Annual_Income_k", "Spending_Score", "Purchase_Frequency", "Recency_Days"]
    
    # Normalize between 0 and 1 for radar comparison
    radar_data = df_filtered.groupby("Cluster_Persona")[radar_cols].mean()
    radar_norm = (radar_data - radar_data.min()) / (radar_data.max() - radar_data.min() + 1e-6)

    fig_radar = go.Figure()
    for persona_name in radar_norm.index:
        values = radar_norm.loc[persona_name].tolist()
        values.append(values[0])
        categories = [c.replace("_", " ") for c in radar_cols]
        categories.append(categories[0])
        fig_radar.add_trace(go.Scatterpolar(
            r=values,
            theta=categories,
            fill='toself',
            name=persona_name
        ))

    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
        showlegend=True,
        title="Normalized Segment Footprint (0 = Lowest, 1 = Highest)",
        height=450,
        margin=dict(l=40, r=40, t=40, b=40)
    )
    st.plotly_chart(fig_radar, use_container_width=True)

# ---------------------------------------------------------
# Tab 4: RFM CRM Segmentation
# ---------------------------------------------------------
with tabs[3]:
    st.markdown("### 🎯 RFM (Recency, Frequency, Monetary) CRM Segmentation")
    st.markdown(
        "Quantile-based behavioral segmentation mapping customer transaction velocity, consistency, and gross revenue."
    )

    col_r1, col_r2 = st.columns([3, 2])

    rfm_table = df_filtered.groupby("RFM_Segment").agg(
        Customers=("Customer_ID", "count"),
        Total_Revenue=("Annual_Spend", "sum"),
        Avg_Recency=("Recency_Days", "mean"),
        Avg_Frequency=("Purchase_Frequency", "mean"),
        Avg_Spend=("Annual_Spend", "mean")
    ).reset_index().sort_values(by="Total_Revenue", ascending=False)

    with col_r1:
        fig_rfm_bar = px.bar(
            rfm_table,
            x="RFM_Segment",
            y="Total_Revenue",
            color="RFM_Segment",
            title="Total Revenue ($) Contribution by RFM Segment",
            text_auto=".2s",
            color_discrete_sequence=px.colors.qualitative.Dark24
        )
        fig_rfm_bar.update_layout(height=380, margin=dict(l=20, r=20, t=40, b=20), showlegend=False)
        st.plotly_chart(fig_rfm_bar, use_container_width=True)

    with col_r2:
        fig_rfm_tree = px.treemap(
            rfm_table,
            path=["RFM_Segment"],
            values="Customers",
            title="Customer Base Treemap by RFM Segment",
            color="Avg_Spend",
            color_continuous_scale="Viridis"
        )
        fig_rfm_tree.update_layout(height=380, margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig_rfm_tree, use_container_width=True)

    st.markdown("#### 📋 RFM Matrix & Action Recommendations")
    rfm_tactics = {
        "Champions": "VIP club, referral incentives, premium concierge, exclusive product drops.",
        "Loyal Customers": "Cross-sell high-value accessories, solicit product reviews, tier upgrades.",
        "Potential Loyalists": "Offer membership rewards, recommend trending products, engagement emails.",
        "New Customers": "Welcome email sequence, onboarding guide, 2nd-order discount voucher.",
        "Need Attention": "Time-limited flash promos, personalized surveys to gauge satisfaction.",
        "At Risk": "High-urgency win-back discounts, cart reminder campaigns, reactivation gifts.",
        "Can't Lose Them": "Direct outreach from account manager, executive feedback calls, deep discount.",
        "Hibernating / Churned": "Automated re-engagement dripper campaign; sunset unresponsive contacts."
    }

    rfm_table["Recommended_Action"] = rfm_table["RFM_Segment"].map(rfm_tactics)
    rfm_display = rfm_table.copy()
    rfm_display["Total_Revenue"] = rfm_display["Total_Revenue"].apply(lambda x: f"${x:,.0f}")
    rfm_display["Avg_Spend"] = rfm_display["Avg_Spend"].apply(lambda x: f"${x:,.1f}")
    rfm_display["Avg_Recency"] = rfm_display["Avg_Recency"].apply(lambda x: f"{x:.1f}d")
    rfm_display["Avg_Frequency"] = rfm_display["Avg_Frequency"].apply(lambda x: f"{x:.1f} orders")
    st.dataframe(rfm_display, use_container_width=True, hide_index=True)

# ---------------------------------------------------------
# Tab 5: Customer Personas & Marketing Playbook
# ---------------------------------------------------------
with tabs[4]:
    st.markdown("### 👤 Customer Persona Profiles & Strategic Marketing Playbook")
    st.markdown("Actionable strategies, communication channels, and product recommendations tailored to each cluster.")

    personas = [
        {
            "name": "👑 Elite VIP Champions",
            "tag": "High Income | High Spending | High Loyalty",
            "color": "#1e40af",
            "desc": "Affluent, brand-loyal professionals who demand premium quality and fast fulfillment. Low price sensitivity.",
            "metrics": "Avg Income: $108k | Avg Spend: $16,200 | Frequency: 36 orders/yr | Recency: 18 days",
            "prefs": "Electronics, Luxury Fashion | Mobile App & Web | Credit Card",
            "strategy": "Deliver bespoke white-glove treatment. Offer VIP access to pre-launches, personal account styling, and exclusive loyalty tier rewards with zero-threshold free shipping.",
            "campaign": "Executive Black Card Loyalty Preview & Private Showroom Access."
        },
        {
            "name": "💎 Affluent Savers (Cautious Wealth)",
            "tag": "High Income | Low Spending | High Evaluation",
            "color": "#0d9488",
            "desc": "High earners with high financial prudence. They research thoroughly and purchase high-ticket items infrequently.",
            "metrics": "Avg Income: $102k | Avg Spend: $2,450 | Frequency: 8 orders/yr | Recency: 110 days",
            "prefs": "Home & Living, High-End Tech | Desktop Web | Credit Card",
            "strategy": "Build trust through product longevity and warranties rather than standard sales pitches. Position premium bundled packages and extended service guarantees.",
            "campaign": "Invest for Life: Premium Home & Tech Durability Guarantees."
        },
        {
            "name": "⚡ Young Trendsetters & Impulse Shoppers",
            "tag": "Moderate Income | High Spending | Mobile-First",
            "color": "#7c3aed",
            "desc": "Young, digitally native consumers driven by viral trends, social media aesthetics, and instant gratification.",
            "metrics": "Avg Income: $46k | Avg Spend: $6,500 | Frequency: 29 orders/yr | Recency: 24 days",
            "prefs": "Fashion, Beauty & Personal Care | Mobile App (75%) | UPI / Wallets",
            "strategy": "Engage via push notifications, Instagram/TikTok influencer collaborations, flash 24-hour drops, and flexible Buy-Now-Pay-Later (BNPL) checkout options.",
            "campaign": "Flash Friday Drops & Trending TikTok Aesthetic Curations."
        },
        {
            "name": "🏷️ Bargain Seekers & Value Shoppers",
            "tag": "Low Income | Low Spending | Highly Price Sensitive",
            "color": "#ea580c",
            "desc": "Cost-conscious shoppers who wait for promotional coupons, holiday clearances, and bulk buying savings.",
            "metrics": "Avg Income: $32k | Avg Spend: $1,150 | Frequency: 7 orders/yr | Recency: 135 days",
            "prefs": "Groceries, Household Essentials | In-Store & Web | Debit Card / UPI",
            "strategy": "Incentivize larger baskets through volume discount bundles ('Buy 2 Get 1 Free') and clear price-match assurances.",
            "campaign": "Super-Saver Weekend: Flat 30% Off Bulk Essentials + Free Delivery."
        },
        {
            "name": "🏡 Core Balanced Families",
            "tag": "Middle Income | Moderate Spending | Predictable Rhythm",
            "color": "#059669",
            "desc": "Family household buyers purchasing predictable quantities of household goods, groceries, and seasonal needs.",
            "metrics": "Avg Income: $62k | Avg Spend: $4,800 | Frequency: 18 orders/yr | Recency: 48 days",
            "prefs": "Groceries, Home & Living, Apparel | Omnichannel | Credit / Debit",
            "strategy": "Enroll in automated monthly replenishment (Subscribe & Save) to lock in lifetime consistency. Offer family-sized bundles and school/holiday promos.",
            "campaign": "Family Pantry Auto-Refill: Save 15% Monthly on Everyday Staples."
        }
    ]

    for p in personas:
        st.markdown(f"""
        <div class="persona-card" style="border-left-color: {p['color']};">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <h3 style="margin:0; color:{p['color']};">{p['name']}</h3>
                <span style="font-size:0.85rem; font-weight:700; color:{p['color']};">{p['tag']}</span>
            </div>
            <p style="margin: 8px 0 4px 0; font-size:0.95rem; color:#334155;">{p['desc']}</p>
            <div style="background:#ffffff; border-radius:8px; padding:10px 14px; margin: 8px 0; border:1px solid #e2e8f0; font-size:0.88rem;">
                <b>📊 Benchmark Stats:</b> {p['metrics']}<br>
                <b>🛍️ Preference Footprint:</b> {p['prefs']}
            </div>
            <div style="font-size:0.9rem; color:#0f172a; margin-top:6px;">
                <b>🚀 Strategic Growth Objective:</b> {p['strategy']}<br>
                <b>🎯 Hero Marketing Campaign:</b> <i>"{p['campaign']}"</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Tab 6: Predict & Segment New Customer (Live Classifier)
# ---------------------------------------------------------
with tabs[5]:
    st.markdown("### 🔮 Classify & Segment a New Customer")
    st.markdown("Input prospective customer attributes to automatically classify them into an ML cluster and generate personalized recommendations.")

    col_inp1, col_inp2 = st.columns(2)

    with col_inp1:
        new_name = st.text_input("Customer Name", "Jordan Taylor")
        new_age = st.slider("Age", min_value=18, max_value=80, value=35)
        new_income = st.slider("Annual Income ($k)", min_value=15, max_value=150, value=95)
        new_score = st.slider("Spending Score (1-100)", min_value=1, max_value=100, value=82)

    with col_inp2:
        new_freq = st.slider("Annual Purchase Frequency (Orders)", min_value=1, max_value=60, value=32)
        new_rec = st.slider("Recency (Days Since Last Order)", min_value=1, max_value=365, value=15)
        new_cat = st.selectbox("Preferred Product Category", ["Electronics", "Apparel & Fashion", "Home & Living", "Groceries & Essentials", "Beauty & Personal Care"])
        new_channel = st.selectbox("Preferred Channel", ["Mobile App", "Website", "In-Store"])

    predict_btn = st.button("🚀 Classify Customer Segment", type="primary")

    if predict_btn:
        # Scale input
        sample_features = np.array([[new_age, new_income, new_score, new_freq, new_rec]])
        sample_scaled = scaler.transform(sample_features)
        
        # Predict cluster
        if algorithm_choice == "K-Means":
            pred_cluster = model.predict(sample_scaled)[0]
        else:
            # Nearest cluster center
            cluster_centers = [X_scaled[cluster_labels == c].mean(axis=0) for c in range(num_clusters)]
            dists = [np.linalg.norm(sample_scaled - center) for center in cluster_centers]
            pred_cluster = int(np.argmin(dists))

        assigned_persona = persona_map.get(pred_cluster, f"Segment {pred_cluster+1}")
        
        # Predict RFM
        r_sc = 5 if new_rec <= 30 else (4 if new_rec <= 60 else (3 if new_rec <= 120 else (2 if new_rec <= 200 else 1)))
        f_sc = 5 if new_freq >= 30 else (4 if new_freq >= 20 else (3 if new_freq >= 12 else (2 if new_freq >= 6 else 1)))
        m_est = new_income * 1000 * (new_score / 100) * 0.15
        m_sc = 5 if m_est >= 10000 else (4 if m_est >= 5000 else (3 if m_est >= 2500 else (2 if m_est >= 1000 else 1)))
        
        row_temp = pd.Series({"R_Score": r_sc, "F_Score": f_sc, "M_Score": m_sc})
        pred_rfm = get_rfm_segment(row_temp)

        st.success(f"✨ Classification Complete for **{new_name}**!")
        
        res_col1, res_col2, res_col3 = st.columns(3)
        with res_col1:
            st.metric("Assigned ML Segment", assigned_persona)
        with res_col2:
            st.metric("CRM RFM Tier", pred_rfm)
        with res_col3:
            st.metric("Estimated Annual Spend", f"${m_est:,.0f}")

        st.markdown(f"""
        <div style="background:#eff6ff; border:1px solid #bfdbfe; border-radius:10px; padding:16px; margin-top:12px;">
            <h4 style="color:#1e40af; margin-top:0;">🎯 Actionable Playbook for {new_name}:</h4>
            <ul>
                <li><b>Primary Communication Channel:</b> Deliver personalized messaging via <b>{new_channel}</b>.</li>
                <li><b>Product Recommendation:</b> Highlight premier arrivals in <b>{new_cat}</b>.</li>
                <li><b>Discount Sensitivity:</b> {'Low (emphasize exclusivity and premium service)' if new_score > 60 else 'High (offer coupon codes and bundle savings)'}.</li>
                <li><b>Lifecycle Strategy:</b> {rfm_tactics.get(pred_rfm, 'Standard customer nurture program.')}</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

# ---------------------------------------------------------
# Footer
# ---------------------------------------------------------
st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.85rem;'>"
    "Customer Analytics & Segmentation Platform | Powered by Python, Scikit-Learn & Streamlit"
    "</div>",
    unsafe_allow_html=True
)

