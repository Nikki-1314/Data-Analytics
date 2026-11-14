# 🎯 Customer Segmentation & Behavioral Analytics Platform

An end-to-end Data Analytics and Unsupervised Machine Learning project designed to segment retail and e-commerce customers based on **demographics** and **purchasing behaviors**.

Featuring an interactive **Streamlit Web Application**, **RFM CRM scoring**, **K-Means & Hierarchical Clustering**, **PCA 2D & 3D visualizations**, a styled **Executive Excel Dashboard**, and a publication-grade **Project Report**.

---

## 🌟 Key Features

1. **Multi-Source Data Ingestion:**
   - 📂 Import data from **CSV** files (`data/customer_segmentation_data.csv`)
   - 📊 Import data from **Excel** workbooks (`data/customer_segmentation_data.xlsx`)
   - 🗄️ Query directly from **SQLite Databases** (`data/customer_segmentation.db`)
   - 📤 File uploaders allowing users to upload custom datasets on the fly!

2. **Unsupervised Machine Learning & Clustering:**
   - **Feature Scaling:** Standardized demographic and behavioral vectors via `StandardScaler`.
   - **Optimal Cluster Validation:** Mathematical confirmation via the **Elbow Method (Inertia/WCSS)** and **Silhouette Coefficient Analysis** ($k=2$ to $k=10$).
   - **Algorithms Implemented:** K-Means Clustering and Agglomerative Hierarchical Clustering with Ward linkage.
   - **Dimensionality Reduction (PCA):** 2D & 3D Principal Component Analysis for high-dimensional visual projection and boundary evaluation.

3. **RFM (Recency, Frequency, Monetary) CRM Segmentation:**
   - Quantile scoring ($1 - 5$) across Recency, Frequency, and Monetary spend.
   - Classification into 8 actionable CRM cohorts: *Champions*, *Loyal Customers*, *Potential Loyalists*, *New Customers*, *Need Attention*, *At Risk*, *Can't Lose Them*, and *Hibernating*.

4. **Interactive Streamlit Web Dashboard (`app.py`):**
   - **Executive Overview:** High-level KPI cards (Total Customers, Annual Revenue, Avg Income, Avg Spending Score, Recency), customer & revenue share comparison bar charts, donut charts, and searchable data viewer.
   - **Exploratory Data Analysis (EDA):** Interactive Plotly histograms, categorical preference breakdowns, and Pearson correlation matrix heatmap.
   - **ML Clustering Lab:** Live slider for $k$ ($2$ to $8$), algorithm switcher (K-Means vs Hierarchical), interactive Elbow & Silhouette curves, 2D scatter plot with centroids, 3D PCA interactive projection, and normalized spider/radar chart.
   - **RFM CRM Segmentation:** Treemap and revenue distribution bar chart with targeted CRM actions.
   - **Customer Personas & Strategic Playbook:** Rich persona cards with demographics, behavioral traits, preferred categories, and hero marketing campaigns.
   - **Predict & Segment New Customer:** Interactive tool to classify a prospective shopper in real-time and generate personalized recommendations.

5. **Deliverables Included:**
   - 🚀 **Interactive Streamlit Web Application** (`app.py`)
   - 📗 **Executive Excel Dashboard with Charts & KPIs** (`reports/Customer_Segmentation_Executive_Dashboard.xlsx`)
   - 📓 **Jupyter Notebook with EDA & Clustering Models** (`notebooks/customer_segmentation_analysis.ipynb`)
   - 📝 **Formal Analytical & Strategic Project Report** (`PROJECT_REPORT.md`)
   - 📊 **Benchmark Dataset across 3 Formats** (CSV, Excel, SQLite in `data/`)

---

## 🚀 How to Run the Project

### 1. Launch the Interactive Web Dashboard:
Run the following command in your terminal:
```bash
python -m streamlit run app.py
```
*(Or `streamlit run app.py` if your Python Scripts directory is in your system PATH).*
This will open your browser at `http://localhost:8501`.

### 2. View the Executive Excel Dashboard:
Open `reports/Customer_Segmentation_Executive_Dashboard.xlsx` in Microsoft Excel. It includes:
- **Executive Dashboard:** Styled navy KPI cards, customer count column chart, and revenue contribution pie chart.
- **Cluster Profiles:** Statistical benchmark matrix with mean age, income, spend, frequency, and strategic focus per cluster.
- **RFM Analysis:** Revenue and customer share table with recommended CRM actions.
- **Customer Master Data:** Formatted records with alternating row colors and number formats.

### 3. Open the Jupyter Notebook:
```bash
python -m notebook notebooks/customer_segmentation_analysis.ipynb
```
Or open the notebook in VS Code or JupyterLab to inspect all step-by-step EDA code, elbow plots, silhouette curves, PCA projections, and hierarchical dendrograms.

---

## 📁 Repository Structure

```
data analytics task 2/
├── app.py                                          # Main Streamlit dashboard application
├── generate_data.py                                # Generates realistic multi-format customer data
├── generate_excel_dashboard.py                     # Generates formatted Excel workbook (.xlsx)
├── generate_notebook.py                            # Generates the Jupyter Notebook (.ipynb)
├── PROJECT_REPORT.md                               # Formal analytical and business report
├── README.md                                       # Project documentation
├── requirements.txt                                # Python package dependencies
├── data/
│   ├── customer_segmentation_data.csv              # Audited customer dataset (CSV)
│   ├── customer_segmentation_data.xlsx             # Benchmark dataset (Excel)
│   └── customer_segmentation.db                    # Relational SQLite database
├── reports/
│   └── Customer_Segmentation_Executive_Dashboard.xlsx # Formatted Excel Dashboard
└── notebooks/
    └── customer_segmentation_analysis.ipynb        # End-to-end EDA & ML Jupyter Notebook
```

---

## 👥 Customer Segment Personas Overview

| Persona | Demographics | Behavioral DNA | Preferred Channel & Category | Strategic Focus |
| :--- | :--- | :--- | :--- | :--- |
| **👑 Elite VIP Champions** | Income: $108k \| Age: 40 | Spend: $16.2k \| Freq: 36/yr \| Recency: 18d | Mobile App & Web \| Electronics & Fashion | **Retention & Exclusivity:** Concierge support, VIP previews, free priority shipping. |
| **💎 Affluent Savers** | Income: $102k \| Age: 52 | Spend: $2.4k \| Freq: 8/yr \| Recency: 110d | Desktop Web \| Home & High-End Tech | **Credibility & Warranties:** Durability guarantees, high-ticket bundles. |
| **⚡ Young Trendsetters** | Income: $46k \| Age: 26 | Spend: $6.5k \| Freq: 29/yr \| Recency: 24d | Mobile App (75%) \| Fashion & Beauty | **Viral & Social Engagement:** Push notifications, TikTok drops, BNPL checkout. |
| **🏷️ Bargain Seekers** | Income: $32k \| Age: 45 | Spend: $1.1k \| Freq: 7/yr \| Recency: 135d | In-Store & Web \| Groceries & Essentials | **Basket Building:** Volume clearance bundles, coupons, free shipping thresholds. |
| **🏡 Core Balanced Families** | Income: $62k \| Age: 42 | Spend: $4.8k \| Freq: 18/yr \| Recency: 48d | Omnichannel \| Groceries & Household | **Subscription Lock-in:** Subscribe & Save refills, family combo packs. |

---

## 🛠️ Installation & Requirements

Install required dependencies using pip:
```bash
pip install -r requirements.txt
```
Dependencies:
- `pandas`
- `numpy`
- `scikit-learn`
- `scipy`
- `matplotlib`
- `seaborn`
- `plotly`
- `streamlit`
- `openpyxl`
- `nbformat`

