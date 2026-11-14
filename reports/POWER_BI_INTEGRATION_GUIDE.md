# 📊 Power BI Customer Segmentation & Analytics Integration Guide

This guide details step-by-step instructions for loading the segmented customer data into **Microsoft Power BI Desktop**, creating interactive dashboard visuals, and utilizing custom DAX measures.

---

## 1. Quick Data Ingestion into Power BI

1. Open **Power BI Desktop**.
2. Click **Get Data** &rarr; **Text/CSV**.
3. Navigate to and select:
   `c:\Users\NIKHITHA\OneDrive\Desktop\data analytics task 2\data\customer_segmentation_data.csv`
4. In the preview window, verify the column headers and click **Transform Data** (Power Query) or **Load**.

---

## 2. Running the Python Clustering Script inside Power BI (Optional Native ML)

Power BI allows executing Python scripts directly within Power Query:
1. In Power Query Editor, go to **Transform** &rarr; **Run Python script**.
2. Paste the following script:
```python
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

features = ["Age", "Annual_Income_k", "Spending_Score", "Purchase_Frequency", "Recency_Days"]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(dataset[features])

kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
dataset['Cluster'] = kmeans.fit_predict(X_scaled)

# Assign Personas
persona_map = {
    0: "Elite VIP Champions",
    1: "Affluent Savers",
    2: "Young Trendsetters",
    3: "Bargain Seekers",
    4: "Core Balanced Families"
}
dataset['Customer_Segment'] = dataset['Cluster'].map(persona_map)
```
3. Click **OK** and then **Close & Apply**.

*(Note: The CSV file already comes pre-configured with all behavioral attributes and segment identifiers ready for immediate visual dragging!)*

---

## 3. Essential DAX Measures for Your Power BI Dashboard

In the **Modeling** ribbon, select **New Measure** and add the following DAX calculations:

```dax
// 1. Total Audited Customers
Total Customers = DISTINCTCOUNT('customer_segmentation_data'[Customer_ID])

// 2. Total Annual Spend ($)
Total Annual Spend = SUM('customer_segmentation_data'[Annual_Spend])

// 3. Average Annual Income ($k)
Average Annual Income = AVERAGE('customer_segmentation_data'[Annual_Income_k])

// 4. Average Spending Score
Average Spending Score = AVERAGE('customer_segmentation_data'[Spending_Score])

// 5. Average Order Value (AOV)
Average Order Value = AVERAGE('customer_segmentation_data'[Average_Order_Value])

// 6. Percentage Share of Total Customer Base
Customer Base Share % = 
DIVIDE(
    [Total Customers],
    CALCULATE([Total Customers], ALL('customer_segmentation_data'))
)

// 7. Percentage Share of Total Enterprise Revenue
Revenue Contribution Share % = 
DIVIDE(
    [Total Annual Spend],
    CALCULATE([Total Annual Spend], ALL('customer_segmentation_data'))
)
```

---

## 4. Recommended Power BI Dashboard Visual Layout

| Visual Type | Location | Fields & Configuration |
| :--- | :--- | :--- |
| **KPI Card 1** | Top Left | `[Total Customers]` (Display units: None) |
| **KPI Card 2** | Top Middle-Left | `[Total Annual Spend]` (Display units: Millions `$M`) |
| **KPI Card 3** | Top Middle-Right | `[Average Annual Income]` |
| **KPI Card 4** | Top Right | `[Average Spending Score]` (out of 100) |
| **Scatter Chart** | Center-Left | **X-Axis:** `Annual_Income_k`, **Y-Axis:** `Spending_Score`, **Legend:** `Customer_Segment`, **Size:** `Annual_Spend` |
| **Donut Chart** | Center-Right | **Legend:** `Customer_Segment`, **Values:** `[Total Annual Spend]` |
| **Clustered Column Chart** | Bottom-Left | **X-Axis:** `Customer_Segment`, **Y-Axis:** `[Customer Base Share %]` and `[Revenue Contribution Share %]` |
| **Matrix / Table** | Bottom-Right | **Rows:** `Customer_Segment`, **Values:** `Preferred_Category`, `Preferred_Channel`, `[Average Order Value]` |
| **Slicers (Top/Side)** | Left Pane | `Region`, `Preferred_Channel`, `Gender`, `Discount_Sensitivity` |

