import os
import sqlite3
import random
import numpy as np
import pandas as pd

def generate_customer_dataset(num_records=2200, seed=42):
    np.random.seed(seed)
    random.seed(seed)

    # Names repository
    first_names = [
        "Aarav", "Emily", "Liam", "Sophia", "Noah", "Olivia", "James", "Emma",
        "Alexander", "Ava", "Benjamin", "Isabella", "Lucas", "Mia", "Henry", "Charlotte",
        "Daniel", "Amelia", "Michael", "Harper", "Rohan", "Priya", "Ananya", "Vikram",
        "Neha", "Rajesh", "Siddharth", "Kavita", "Aditya", "Sneha", "Mateo", "Camila",
        "Santiago", "Valentina", "Leonardo", "Elena", "Marcus", "Chloe", "Kai", "Zara",
        "Ethan", "Zoe", "Oliver", "Grace", "David", "Lily", "John", "Ella"
    ]
    last_names = [
        "Patel", "Johnson", "Smith", "Garcia", "Miller", "Davis", "Rodriguez", "Martinez",
        "Hernandez", "Lopez", "Gonzalez", "Wilson", "Anderson", "Thomas", "Taylor", "Moore",
        "Jackson", "Martin", "Lee", "Perez", "Sharma", "Verma", "Rao", "Gupta",
        "Mehta", "Reddy", "Nair", "Iyer", "Chopra", "Deshmukh", "Silva", "Santos",
        "Kim", "Chen", "Tanaka", "Watanabe", "Muller", "Schmidt", "Dubois", "Leroy"
    ]

    cities_by_region = {
        "North": ["New York", "Chicago", "Boston", "Minneapolis", "Detroit"],
        "South": ["Atlanta", "Miami", "Dallas", "Houston", "Austin"],
        "East": ["Philadelphia", "Washington D.C.", "Baltimore", "Pittsburgh", "Newark"],
        "West": ["Los Angeles", "San Francisco", "Seattle", "Denver", "Phoenix"]
    }

    categories = ["Electronics", "Apparel & Fashion", "Home & Living", "Groceries & Essentials", "Beauty & Personal Care"]
    channels = ["Mobile App", "Website", "In-Store"]
    payment_methods = ["Credit Card", "UPI / Digital Wallet", "Debit Card", "Net Banking"]
    education_levels = ["High School", "Bachelor's", "Master's", "Doctorate"]
    marital_statuses = ["Single", "Married", "Divorced"]

    # We will generate 5 realistic behavioral archetypes / natural clusters
    # 1. Champions (Elite VIPs): High Income, High Spending, High Frequency, Low Recency
    # 2. Affluent Savers (Cautious Wealth): High Income, Low Spending, Low Frequency, Moderate Recency
    # 3. Young Trendsetters: Moderate Income, High Spending Score, Mobile App heavy, Fashion/Tech
    # 4. Budget Bargain Hunters: Low-to-Moderate Income, Low-to-Moderate Spend, High Discount Sensitivity
    # 5. Core Balanced Families: Middle Income, Average Spend, Groceries/Home, Balanced Frequency

    cluster_proportions = [0.22, 0.18, 0.24, 0.20, 0.16]
    cluster_counts = [int(p * num_records) for p in cluster_proportions]
    cluster_counts[-1] = num_records - sum(cluster_counts[:-1])

    records = []
    cust_id_counter = 1001

    for c_idx, count in enumerate(cluster_counts):
        for _ in range(count):
            first = random.choice(first_names)
            last = random.choice(last_names)
            customer_name = f"{first} {last}"
            gender = np.random.choice(["Female", "Male", "Non-Binary"], p=[0.51, 0.46, 0.03])
            region = random.choice(list(cities_by_region.keys()))
            city = random.choice(cities_by_region[region])
            education = np.random.choice(education_levels, p=[0.20, 0.52, 0.22, 0.06])
            marital = np.random.choice(marital_statuses, p=[0.38, 0.50, 0.12])

            if c_idx == 0:
                # 1. Champions / VIPs
                age = int(np.clip(np.random.normal(40, 8), 26, 62))
                annual_income_k = round(float(np.clip(np.random.normal(108, 16), 78, 148)), 1)
                spending_score = int(np.clip(np.random.normal(84, 8), 68, 99))
                purchase_freq = int(np.clip(np.random.normal(36, 7), 22, 58))
                recency_days = int(np.clip(np.random.normal(18, 10), 2, 45))
                pref_cat = np.random.choice(categories, p=[0.42, 0.30, 0.15, 0.05, 0.08])
                pref_channel = np.random.choice(channels, p=[0.45, 0.40, 0.15])
                pref_payment = np.random.choice(payment_methods, p=[0.65, 0.22, 0.08, 0.05])
                discount_sensitivity = np.random.choice(["Low", "Medium", "High"], p=[0.75, 0.20, 0.05])
                web_visits = int(np.clip(np.random.normal(28, 6), 14, 48))
                return_rate = round(float(np.clip(np.random.normal(5.2, 2.0), 1.0, 12.0)), 1)
                tenure = int(np.clip(np.random.normal(46, 14), 12, 72))
                tier = np.random.choice(["Platinum", "Gold"], p=[0.70, 0.30])
                annual_spend = round(annual_income_k * 1000 * (spending_score / 100) * np.random.uniform(0.14, 0.22), 2)

            elif c_idx == 1:
                # 2. Affluent Savers / Cautious High-Earners
                age = int(np.clip(np.random.normal(52, 10), 34, 70))
                annual_income_k = round(float(np.clip(np.random.normal(102, 18), 75, 145)), 1)
                spending_score = int(np.clip(np.random.normal(24, 7), 6, 42))
                purchase_freq = int(np.clip(np.random.normal(8, 3), 2, 16))
                recency_days = int(np.clip(np.random.normal(110, 35), 45, 230))
                pref_cat = np.random.choice(categories, p=[0.35, 0.12, 0.32, 0.16, 0.05])
                pref_channel = np.random.choice(channels, p=[0.20, 0.50, 0.30])
                pref_payment = np.random.choice(payment_methods, p=[0.50, 0.20, 0.15, 0.15])
                discount_sensitivity = np.random.choice(["Low", "Medium", "High"], p=[0.40, 0.45, 0.15])
                web_visits = int(np.clip(np.random.normal(9, 3), 2, 20))
                return_rate = round(float(np.clip(np.random.normal(4.0, 1.5), 0.5, 9.0)), 1)
                tenure = int(np.clip(np.random.normal(36, 15), 6, 68))
                tier = np.random.choice(["Silver", "Gold", "Bronze"], p=[0.50, 0.30, 0.20])
                annual_spend = round(annual_income_k * 1000 * (spending_score / 100) * np.random.uniform(0.06, 0.12), 2)

            elif c_idx == 2:
                # 3. Young Trendsetters & Impulse Shoppers
                age = int(np.clip(np.random.normal(26, 4), 18, 35))
                annual_income_k = round(float(np.clip(np.random.normal(46, 10), 24, 68)), 1)
                spending_score = int(np.clip(np.random.normal(85, 7), 68, 99))
                purchase_freq = int(np.clip(np.random.normal(29, 6), 16, 48))
                recency_days = int(np.clip(np.random.normal(24, 12), 3, 60))
                pref_cat = np.random.choice(categories, p=[0.25, 0.45, 0.05, 0.05, 0.20])
                pref_channel = np.random.choice(channels, p=[0.75, 0.20, 0.05])
                pref_payment = np.random.choice(payment_methods, p=[0.30, 0.55, 0.10, 0.05])
                discount_sensitivity = np.random.choice(["Low", "Medium", "High"], p=[0.15, 0.55, 0.30])
                web_visits = int(np.clip(np.random.normal(32, 7), 16, 55))
                return_rate = round(float(np.clip(np.random.normal(12.5, 3.5), 4.0, 26.0)), 1)
                tenure = int(np.clip(np.random.normal(20, 9), 3, 42))
                tier = np.random.choice(["Silver", "Gold", "Bronze"], p=[0.45, 0.35, 0.20])
                annual_spend = round(annual_income_k * 1000 * (spending_score / 100) * np.random.uniform(0.12, 0.19), 2)

            elif c_idx == 3:
                # 4. Budget Bargain Hunters / Value Shoppers
                age = int(np.clip(np.random.normal(45, 12), 20, 71))
                annual_income_k = round(float(np.clip(np.random.normal(32, 8), 16, 48)), 1)
                spending_score = int(np.clip(np.random.normal(26, 8), 5, 45))
                purchase_freq = int(np.clip(np.random.normal(7, 3), 2, 14))
                recency_days = int(np.clip(np.random.normal(135, 45), 40, 310))
                pref_cat = np.random.choice(categories, p=[0.08, 0.18, 0.22, 0.44, 0.08])
                pref_channel = np.random.choice(channels, p=[0.25, 0.35, 0.40])
                pref_payment = np.random.choice(payment_methods, p=[0.18, 0.35, 0.37, 0.10])
                discount_sensitivity = np.random.choice(["Low", "Medium", "High"], p=[0.05, 0.25, 0.70])
                web_visits = int(np.clip(np.random.normal(8, 3), 2, 18))
                return_rate = round(float(np.clip(np.random.normal(6.0, 2.0), 1.0, 14.0)), 1)
                tenure = int(np.clip(np.random.normal(26, 12), 2, 55))
                tier = np.random.choice(["Bronze", "Silver"], p=[0.82, 0.18])
                annual_spend = round(annual_income_k * 1000 * (spending_score / 100) * np.random.uniform(0.07, 0.14), 2)

            else:
                # 5. Core Balanced Families
                age = int(np.clip(np.random.normal(42, 8), 28, 62))
                annual_income_k = round(float(np.clip(np.random.normal(62, 11), 45, 84)), 1)
                spending_score = int(np.clip(np.random.normal(51, 8), 36, 66))
                purchase_freq = int(np.clip(np.random.normal(18, 4), 9, 28))
                recency_days = int(np.clip(np.random.normal(48, 18), 12, 115))
                pref_cat = np.random.choice(categories, p=[0.20, 0.20, 0.26, 0.24, 0.10])
                pref_channel = np.random.choice(channels, p=[0.35, 0.40, 0.25])
                pref_payment = np.random.choice(payment_methods, p=[0.40, 0.30, 0.20, 0.10])
                discount_sensitivity = np.random.choice(["Low", "Medium", "High"], p=[0.20, 0.60, 0.20])
                web_visits = int(np.clip(np.random.normal(16, 4), 6, 28))
                return_rate = round(float(np.clip(np.random.normal(7.0, 2.0), 1.5, 15.0)), 1)
                tenure = int(np.clip(np.random.normal(38, 14), 8, 65))
                tier = np.random.choice(["Silver", "Gold", "Bronze"], p=[0.55, 0.30, 0.15])
                annual_spend = round(annual_income_k * 1000 * (spending_score / 100) * np.random.uniform(0.10, 0.16), 2)

            avg_order_val = round(annual_spend / max(1, purchase_freq), 2)
            # Projected CLV based on tenure, annual spend, and retention rate factor
            clv = round(annual_spend * (1 + (tenure / 24)) * (1 - (return_rate / 100)), 2)

            records.append({
                "Customer_ID": f"CUST-{cust_id_counter}",
                "Customer_Name": customer_name,
                "Gender": gender,
                "Age": age,
                "Annual_Income_k": annual_income_k,
                "Spending_Score": spending_score,
                "Annual_Spend": annual_spend,
                "Purchase_Frequency": purchase_freq,
                "Recency_Days": recency_days,
                "Average_Order_Value": avg_order_val,
                "Preferred_Category": pref_cat,
                "Preferred_Channel": pref_channel,
                "Payment_Method": pref_payment,
                "Discount_Sensitivity": discount_sensitivity,
                "Return_Rate_Pct": return_rate,
                "Website_Visits_Per_Month": web_visits,
                "Customer_Tenure_Months": tenure,
                "Membership_Tier": tier,
                "Marital_Status": marital,
                "Education": education,
                "City": city,
                "Region": region,
                "Customer_Lifetime_Value": clv
            })
            cust_id_counter += 1

    # Shuffle the dataset to avoid ordered cluster bias
    random.shuffle(records)
    df = pd.DataFrame(records)

    # Save to CSV
    os.makedirs("data", exist_ok=True)
    csv_file = os.path.join("data", "customer_segmentation_data.csv")
    df.to_csv(csv_file, index=False)
    print(f"Exported CSV: {csv_file} ({len(df)} records)")

    # Save to Excel
    xlsx_file = os.path.join("data", "customer_segmentation_data.xlsx")
    df.to_excel(xlsx_file, index=False, engine="openpyxl")
    print(f"Exported Excel: {xlsx_file}")

    # Save to SQLite Database
    db_file = os.path.join("data", "customer_segmentation.db")
    conn = sqlite3.connect(db_file)
    df.to_sql("customers", conn, if_exists="replace", index=False)
    conn.commit()
    conn.close()
    print(f"Exported SQLite DB: {db_file}")

    return df

if __name__ == "__main__":
    generate_customer_dataset()

