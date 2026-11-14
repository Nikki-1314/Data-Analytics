import os
import openpyxl
from openpyxl.styles import Font, PatternFill, Alignment, Border, Side
from openpyxl.utils import get_column_letter
from openpyxl.chart import BarChart, PieChart, Reference
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

def generate_excel_dashboard():
    csv_path = os.path.join("data", "customer_segmentation_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Missing {csv_path}")

    df = pd.read_csv(csv_path)

    # Perform standard clustering to have cluster labels
    cluster_features = ["Age", "Annual_Income_k", "Spending_Score", "Purchase_Frequency", "Recency_Days"]
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(df[cluster_features])

    kmeans = KMeans(n_clusters=5, random_state=42, n_init=10)
    df["Cluster"] = kmeans.fit_predict(scaled_data)

    # Map clusters to business personas based on their average income and spending score
    cluster_stats = df.groupby("Cluster")[["Annual_Income_k", "Spending_Score", "Annual_Spend"]].mean()
    persona_mapping = {}
    for c_id, row in cluster_stats.iterrows():
        inc = row["Annual_Income_k"]
        score = row["Spending_Score"]
        if inc > 75 and score > 65:
            persona_mapping[c_id] = "Elite VIP Champions"
        elif inc > 75 and score <= 45:
            persona_mapping[c_id] = "Affluent Savers"
        elif inc <= 55 and score > 65:
            persona_mapping[c_id] = "Young Trendsetters"
        elif inc <= 50 and score <= 45:
            persona_mapping[c_id] = "Bargain Seekers"
        else:
            persona_mapping[c_id] = "Core Balanced Families"

    df["Customer_Segment"] = df["Cluster"].map(persona_mapping)

    # RFM Scoring
    df["R_Score"] = pd.qcut(df["Recency_Days"], 5, labels=[5, 4, 3, 2, 1]).astype(int)
    df["F_Score"] = pd.qcut(df["Purchase_Frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df["M_Score"] = pd.qcut(df["Annual_Spend"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5]).astype(int)
    df["RFM_Score"] = df["R_Score"].astype(str) + df["F_Score"].astype(str) + df["M_Score"].astype(str)

    def assign_rfm_segment(row):
        r, f, m = row["R_Score"], row["F_Score"], row["M_Score"]
        if r >= 4 and f >= 4 and m >= 4:
            return "Champions"
        elif f >= 4 and m >= 3:
            return "Loyal Customers"
        elif r >= 4 and f in [2, 3]:
            return "Potential Loyalists"
        elif r >= 4 and f <= 2:
            return "New / Recent Customers"
        elif r in [2, 3] and f in [2, 3]:
            return "Need Attention"
        elif r <= 2 and f >= 3:
            return "At Risk"
        elif r == 1 and f >= 4:
            return "Can't Lose Them"
        else:
            return "Hibernating / Churned"

    df["RFM_Segment"] = df.apply(assign_rfm_segment, axis=1)

    # Initialize openpyxl workbook
    wb = openpyxl.Workbook()

    # Sheet 1: Executive Dashboard
    ws_dash = wb.active
    ws_dash.title = "Executive Dashboard"
    ws_dash.views.sheetView[0].showGridLines = True

    # Styling Palettes
    navy_header_fill = PatternFill(start_color="0F172A", end_color="0F172A", fill_type="solid")
    primary_header_fill = PatternFill(start_color="1E3A8A", end_color="1E3A8A", fill_type="solid")
    card_fill = PatternFill(start_color="F8FAFC", end_color="F8FAFC", fill_type="solid")
    accent_bar_fill = PatternFill(start_color="3B82F6", end_color="3B82F6", fill_type="solid")
    table_header_fill = PatternFill(start_color="2563EB", end_color="2563EB", fill_type="solid")
    alt_row_fill = PatternFill(start_color="F1F5F9", end_color="F1F5F9", fill_type="solid")
    highlight_fill = PatternFill(start_color="EFF6FF", end_color="EFF6FF", fill_type="solid")

    title_font = Font(name="Segoe UI", size=18, bold=True, color="FFFFFF")
    subtitle_font = Font(name="Segoe UI", size=10, italic=True, color="94A3B8")
    card_label_font = Font(name="Segoe UI", size=9, bold=True, color="64748B")
    card_value_font = Font(name="Segoe UI", size=18, bold=True, color="0F172A")
    card_sub_font = Font(name="Segoe UI", size=8, bold=True, color="16A34A")
    table_header_font = Font(name="Segoe UI", size=10, bold=True, color="FFFFFF")
    cell_font = Font(name="Segoe UI", size=9)
    bold_cell_font = Font(name="Segoe UI", size=9, bold=True)

    thin_border = Border(
        left=Side(style='thin', color='CBD5E1'),
        right=Side(style='thin', color='CBD5E1'),
        top=Side(style='thin', color='CBD5E1'),
        bottom=Side(style='thin', color='CBD5E1')
    )

    # 1. Main Banner
    ws_dash.merge_cells("B2:M2")
    ws_dash["B2"] = "CUSTOMER SEGMENTATION & ANALYTICS EXECUTIVE DASHBOARD"
    ws_dash["B2"].font = title_font
    ws_dash["B2"].fill = navy_header_fill
    ws_dash["B2"].alignment = Alignment(horizontal="center", vertical="center")
    ws_dash.row_dimensions[2].height = 40

    ws_dash.merge_cells("B3:M3")
    ws_dash["B3"] = "Machine Learning Segmentation | RFM Behavioral Modeling | Targeted Strategic Growth Playbook"
    ws_dash["B3"].font = subtitle_font
    ws_dash["B3"].fill = navy_header_fill
    ws_dash["B3"].alignment = Alignment(horizontal="center", vertical="center")
    ws_dash.row_dimensions[3].height = 20

    # 2. KPI Cards
    total_customers = len(df)
    total_spend = df["Annual_Spend"].sum()
    avg_income = df["Annual_Income_k"].mean()
    avg_spend_score = df["Spending_Score"].mean()
    avg_recency = df["Recency_Days"].mean()
    avg_aov = df["Average_Order_Value"].mean()

    kpis = [
        ("TOTAL CUSTOMERS", f"{total_customers:,.0f}", "Audited Base", "B", "C"),
        ("TOTAL ANNUAL SPEND", f"${total_spend/1e6:,.2f}M", f"Avg ${(total_spend/total_customers):,.0f}/cust", "D", "E"),
        ("AVG ANNUAL INCOME", f"${avg_income:,.1f}k", "Household Base", "F", "G"),
        ("AVG SPEND SCORE", f"{avg_spend_score:.1f} / 100", "Engagement Index", "H", "I"),
        ("AVG RECENCY", f"{avg_recency:.1f} Days", "Active Cycle", "J", "K"),
        ("AVG ORDER VALUE", f"${avg_aov:.1f}", "Per Basket", "L", "M")
    ]

    for title, val, sub, start_col, end_col in kpis:
        range_str = f"{start_col}5:{end_col}7"
        ws_dash.merge_cells(range_str)
        top_cell = ws_dash[f"{start_col}5"]
        top_cell.value = f"{title}\n{val}\n{sub}"
        top_cell.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        top_cell.font = card_value_font
        
        for row in range(5, 8):
            for col_letter in [start_col, end_col]:
                c = ws_dash[f"{col_letter}{row}"]
                c.fill = card_fill
                c.border = thin_border

    # 3. Cluster Summary Table on Dashboard
    ws_dash["B9"] = "CUSTOMER SEGMENT DISTRIBUTION & REVENUE CONTRIBUTION"
    ws_dash["B9"].font = Font(name="Segoe UI", size=11, bold=True, color="1E3A8A")

    seg_summary = df.groupby("Customer_Segment").agg(
        Count=("Customer_ID", "count"),
        Total_Spend=("Annual_Spend", "sum"),
        Avg_Income=("Annual_Income_k", "mean"),
        Avg_Score=("Spending_Score", "mean"),
        Avg_Recency=("Recency_Days", "mean"),
        Avg_Freq=("Purchase_Frequency", "mean")
    ).reset_index()

    seg_summary["Spend_Share"] = (seg_summary["Total_Spend"] / total_spend) * 100

    dash_headers = ["Segment Name", "Customer Count", "% Base", "Total Revenue ($)", "Revenue Share %", "Avg Income ($k)", "Avg Spend Score", "Avg Recency (Days)"]
    for c_i, h in enumerate(dash_headers, start=2):
        cell = ws_dash.cell(row=10, column=c_i, value=h)
        cell.font = table_header_font
        cell.fill = table_header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border

    for r_i, row in seg_summary.iterrows():
        cur_row = 11 + r_i
        ws_dash.cell(row=cur_row, column=2, value=row["Customer_Segment"]).alignment = Alignment(horizontal="left")
        ws_dash.cell(row=cur_row, column=3, value=row["Count"]).number_format = "#,##0"
        ws_dash.cell(row=cur_row, column=4, value=row["Count"] / total_customers).number_format = "0.0%"
        ws_dash.cell(row=cur_row, column=5, value=row["Total_Spend"]).number_format = "$#,##0"
        ws_dash.cell(row=cur_row, column=6, value=row["Spend_Share"] / 100).number_format = "0.0%"
        ws_dash.cell(row=cur_row, column=7, value=round(row["Avg_Income"], 1)).number_format = "$#,##0.0"
        ws_dash.cell(row=cur_row, column=8, value=round(row["Avg_Score"], 1)).number_format = "0.0"
        ws_dash.cell(row=cur_row, column=9, value=round(row["Avg_Recency"], 1)).number_format = "0.0"

        for col in range(2, 10):
            c = ws_dash.cell(row=cur_row, column=col)
            c.font = cell_font
            c.border = thin_border
            if r_i % 2 == 1:
                c.fill = alt_row_fill

    # 4. Insert Charts on Dashboard
    # Bar Chart for Customer Count by Segment
    chart_bar = BarChart()
    chart_bar.type = "col"
    chart_bar.style = 10
    chart_bar.title = "Customer Count by Segment"
    chart_bar.y_axis.title = "Customers"
    chart_bar.x_axis.title = "Segment"
    chart_bar.height = 10
    chart_bar.width = 16

    data_ref = Reference(ws_dash, min_col=3, min_row=10, max_row=10 + len(seg_summary))
    cats_ref = Reference(ws_dash, min_col=2, min_row=11, max_row=10 + len(seg_summary))
    chart_bar.add_data(data_ref, titles_from_data=True)
    chart_bar.set_categories(cats_ref)
    chart_bar.legend = None
    ws_dash.add_chart(chart_bar, "B18")

    # Pie Chart for Revenue Contribution
    chart_pie = PieChart()
    chart_pie.title = "Revenue Contribution by Segment"
    chart_pie.height = 10
    chart_pie.width = 14
    pie_data = Reference(ws_dash, min_col=5, min_row=10, max_row=10 + len(seg_summary))
    chart_pie.add_data(pie_data, titles_from_data=True)
    chart_pie.set_categories(cats_ref)
    ws_dash.add_chart(chart_pie, "H18")

    # 5. Sheet 2: Cluster Profiles & Action Matrix
    ws_profile = wb.create_sheet(title="Cluster Profiles")
    ws_profile.views.sheetView[0].showGridLines = True

    ws_profile.merge_cells("A1:K1")
    ws_profile["A1"] = "DETAILED CUSTOMER CLUSTER PROFILES & STRATEGIC PLAYBOOK"
    ws_profile["A1"].font = title_font
    ws_profile["A1"].fill = primary_header_fill
    ws_profile["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws_profile.row_dimensions[1].height = 35

    profile_headers = [
        "Segment Persona", "Size (#)", "Share (%)", "Avg Age", "Avg Income ($k)",
        "Spending Score", "Avg Spend ($)", "Avg Frequency", "Avg Recency",
        "Top Category", "Strategic Focus & Action Plan"
    ]
    for c_i, h in enumerate(profile_headers, start=1):
        cell = ws_profile.cell(row=3, column=c_i, value=h)
        cell.font = table_header_font
        cell.fill = table_header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border

    strategy_dict = {
        "Elite VIP Champions": "VIP Concierge, early product drops, loyalty tiers, premium upselling.",
        "Affluent Savers": "Exclusive value bundles, wealth advisory/quality guarantees, re-engagement incentives.",
        "Young Trendsetters": "Mobile app push notifications, social influencer collabs, flash discounts, buy-now-pay-later.",
        "Bargain Seekers": "Budget clearance alerts, coupon codes, bulk savings promotions, price-match guarantees.",
        "Core Balanced Families": "Subscription replenishment, household family bundles, seasonal back-to-school/holiday promos."
    }

    full_stats = df.groupby("Customer_Segment").agg(
        Count=("Customer_ID", "count"),
        Avg_Age=("Age", "mean"),
        Avg_Income=("Annual_Income_k", "mean"),
        Avg_Score=("Spending_Score", "mean"),
        Avg_Spend=("Annual_Spend", "mean"),
        Avg_Freq=("Purchase_Frequency", "mean"),
        Avg_Recency=("Recency_Days", "mean"),
        Top_Category=("Preferred_Category", lambda x: x.mode()[0])
    ).reset_index()

    for r_i, row in full_stats.iterrows():
        cur_row = 4 + r_i
        seg_name = row["Customer_Segment"]
        ws_profile.cell(row=cur_row, column=1, value=seg_name).alignment = Alignment(horizontal="left")
        ws_profile.cell(row=cur_row, column=2, value=row["Count"]).number_format = "#,##0"
        ws_profile.cell(row=cur_row, column=3, value=row["Count"] / total_customers).number_format = "0.0%"
        ws_profile.cell(row=cur_row, column=4, value=round(row["Avg_Age"], 1)).number_format = "0.0"
        ws_profile.cell(row=cur_row, column=5, value=round(row["Avg_Income"], 1)).number_format = "$#,##0.0"
        ws_profile.cell(row=cur_row, column=6, value=round(row["Avg_Score"], 1)).number_format = "0.0"
        ws_profile.cell(row=cur_row, column=7, value=round(row["Avg_Spend"], 2)).number_format = "$#,##0.00"
        ws_profile.cell(row=cur_row, column=8, value=round(row["Avg_Freq"], 1)).number_format = "0.0"
        ws_profile.cell(row=cur_row, column=9, value=round(row["Avg_Recency"], 1)).number_format = "0.0"
        ws_profile.cell(row=cur_row, column=10, value=row["Top_Category"]).alignment = Alignment(horizontal="center")
        ws_profile.cell(row=cur_row, column=11, value=strategy_dict.get(seg_name, "Standard Retention")).alignment = Alignment(horizontal="left")

        for col in range(1, 12):
            c = ws_profile.cell(row=cur_row, column=col)
            c.font = cell_font
            c.border = thin_border
            if r_i % 2 == 1:
                c.fill = alt_row_fill

    # 6. Sheet 3: RFM CRM Analysis
    ws_rfm = wb.create_sheet(title="RFM Analysis")
    ws_rfm.views.sheetView[0].showGridLines = True

    ws_rfm.merge_cells("A1:G1")
    ws_rfm["A1"] = "BEHAVIORAL RFM (RECENCY, FREQUENCY, MONETARY) SEGMENTATION"
    ws_rfm["A1"].font = title_font
    ws_rfm["A1"].fill = navy_header_fill
    ws_rfm["A1"].alignment = Alignment(horizontal="center", vertical="center")
    ws_rfm.row_dimensions[1].height = 35

    rfm_headers = ["RFM Segment", "Customer Count", "% Total Base", "Total Revenue ($)", "Avg Recency (Days)", "Avg Frequency", "Recommended CRM Tactic"]
    for c_i, h in enumerate(rfm_headers, start=1):
        cell = ws_rfm.cell(row=3, column=c_i, value=h)
        cell.font = table_header_font
        cell.fill = table_header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border

    rfm_actions = {
        "Champions": "VIP program, referral bonuses, reward loyalty, high-touch support.",
        "Loyal Customers": "Cross-sell high-value accessories, ask for reviews, recurring incentives.",
        "Potential Loyalists": "Offer membership benefits, recommend trending items, onboarding emails.",
        "New / Recent Customers": "Welcome series, clear onboarding, starter discounts on 2nd purchase.",
        "Need Attention": "Time-sensitive flash offers, surveys, personalized re-engagement.",
        "At Risk": "Aggressive win-back discounts, reminders of saved cart/wishlist items.",
        "Can't Lose Them": "Direct phone outreach, deep discounts, executive survey to resolve pain points.",
        "Hibernating / Churned": "Low-cost automated email reactivation or channel cleanup."
    }

    rfm_summary = df.groupby("RFM_Segment").agg(
        Count=("Customer_ID", "count"),
        Total_Spend=("Annual_Spend", "sum"),
        Avg_Recency=("Recency_Days", "mean"),
        Avg_Freq=("Purchase_Frequency", "mean")
    ).reset_index().sort_values(by="Total_Spend", ascending=False)

    for r_i, row in rfm_summary.iterrows():
        cur_row = 4 + list(rfm_summary.index).index(r_i)
        rfm_name = row["RFM_Segment"]
        ws_rfm.cell(row=cur_row, column=1, value=rfm_name).alignment = Alignment(horizontal="left")
        ws_rfm.cell(row=cur_row, column=2, value=row["Count"]).number_format = "#,##0"
        ws_rfm.cell(row=cur_row, column=3, value=row["Count"] / total_customers).number_format = "0.0%"
        ws_rfm.cell(row=cur_row, column=4, value=row["Total_Spend"]).number_format = "$#,##0"
        ws_rfm.cell(row=cur_row, column=5, value=round(row["Avg_Recency"], 1)).number_format = "0.0"
        ws_rfm.cell(row=cur_row, column=6, value=round(row["Avg_Freq"], 1)).number_format = "0.0"
        ws_rfm.cell(row=cur_row, column=7, value=rfm_actions.get(rfm_name, "Standard Nurture")).alignment = Alignment(horizontal="left")

        for col in range(1, 8):
            c = ws_rfm.cell(row=cur_row, column=col)
            c.font = cell_font
            c.border = thin_border
            if cur_row % 2 == 1:
                c.fill = alt_row_fill

    # 7. Sheet 4: Customer Master Data (First 1,000 records to keep workbook fast)
    ws_data = wb.create_sheet(title="Customer Master Data")
    ws_data.views.sheetView[0].showGridLines = True

    export_cols = [
        "Customer_ID", "Customer_Name", "Gender", "Age", "Annual_Income_k",
        "Spending_Score", "Annual_Spend", "Purchase_Frequency", "Recency_Days",
        "Average_Order_Value", "Preferred_Category", "Preferred_Channel",
        "Payment_Method", "Discount_Sensitivity", "City", "Region", "Customer_Segment", "RFM_Segment"
    ]

    for c_i, h in enumerate(export_cols, start=1):
        cell = ws_data.cell(row=1, column=c_i, value=h.replace("_", " "))
        cell.font = table_header_font
        cell.fill = navy_header_fill
        cell.alignment = Alignment(horizontal="center", vertical="center")
        cell.border = thin_border

    sample_df = df[export_cols].head(1000)
    for r_i, (_, row) in enumerate(sample_df.iterrows(), start=2):
        for c_i, col in enumerate(export_cols, start=1):
            val = row[col]
            cell = ws_data.cell(row=r_i, column=c_i, value=val)
            cell.font = cell_font
            cell.border = thin_border

            if col in ["Annual_Spend", "Average_Order_Value"]:
                cell.number_format = "$#,##0.00"
            elif col in ["Annual_Income_k"]:
                cell.number_format = "$#,##0.0"
            elif col in ["Purchase_Frequency", "Recency_Days", "Spending_Score", "Age"]:
                cell.number_format = "#,##0"

            if r_i % 2 == 1:
                cell.fill = alt_row_fill

    # Auto-adjust column widths for all sheets
    for ws in [ws_dash, ws_profile, ws_rfm, ws_data]:
        for col in ws.columns:
            max_len = 0
            col_letter = get_column_letter(col[0].column)
            for cell in col:
                val_str = str(cell.value or "")
                if "\n" in val_str:
                    val_str = max(val_str.split("\n"), key=len)
                if len(val_str) > max_len:
                    max_len = len(val_str)
            ws.column_dimensions[col_letter].width = max(max_len + 3, 12)

    os.makedirs("reports", exist_ok=True)
    report_file = os.path.join("reports", "Customer_Segmentation_Executive_Dashboard.xlsx")
    wb.save(report_file)
    print(f"Generated Executive Excel Dashboard: {report_file}")

if __name__ == "__main__":
    generate_excel_dashboard()

