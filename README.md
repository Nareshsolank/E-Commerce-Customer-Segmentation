# E-Commerce-Customer-Segmentation
Customer segmentation for an online retail dataset using RFM analysis and K-Means clustering.
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os
from datetime import timedelta

# --- Phase 1: Data Acquisition and Initial Understanding ---

print("--- Phase 1: Data Acquisition and Initial Understanding ---")

# 1. Data Source Identification and Download
# Download the dataset using KaggleHub
# This assumes 'lakshmi25npathi/online-retail-dataset' is correctly configured in your Kaggle API.
try:
    path = kagglehub.dataset_download("lakshmi25npathi/online-retail-dataset")
    print(f"Path to dataset files: {path}")
    # List files in the directory to find the .xlsx or .csv file
    print(f"Files in dataset directory: {os.listdir(path)}")
    file_path = f'{path}/online_retail_II.xlsx' # Assuming it's the Excel file as per your original code
    df = pd.read_excel(file_path)
    print("\nDataset loaded successfully.")
except Exception as e:
    print(f"Error downloading or loading dataset: {e}")
    print("Please ensure you have configured Kaggle API credentials correctly or manually download 'online_retail_II.xlsx' and place it in the same directory as this script.")
    # Fallback for local testing if KaggleHub fails:
    # df = pd.read_excel('online_retail_II.xlsx')


# 2. Simulate SQL Initial Exploration (using Pandas)
print("\nInitial Data Head:")
print(df.head()) # Using print for broader compatibility

print("\nData Info (initial):")
df.info()

print("\nDescriptive Statistics (initial):")
print(df.describe())

print("\nUnique Countries (simulated SELECT DISTINCT Country):")
print(df['Country'].unique())

print(f"\nTotal number of rows (simulated SELECT COUNT(*)): {df.shape[0]}")

print(f"Number of distinct Customer IDs (simulated SELECT COUNT(DISTINCT CustomerID)): {df['Customer ID'].nunique()}")

print(f"Count of negative Quantities: {(df['Quantity'] < 0).sum()}")
print(f"Count of negative Prices: {(df['Price'] < 0).sum()}")


# --- Phase 2: Data Cleaning and Preprocessing ---

print("\n--- Phase 2: Data Cleaning and Preprocessing ---")

# 1. Handle Missing Customer IDs
number_of_rows_before_null_cid_removal = df.shape[0]
df.dropna(subset=['Customer ID'], inplace=True)
number_of_rows_after_null_cid_removal = df.shape[0]
print(f"Number of rows before removing null Customer IDs: {number_of_rows_before_null_cid_removal}")
print(f"Number of rows after removing null Customer IDs: {number_of_rows_after_null_cid_removal}")
print(f"Rows removed due to null Customer IDs: {number_of_rows_before_null_cid_removal - number_of_rows_after_null_cid_removal}")

# 2. Handle Negative Quantities (typically returns, exclude for sales analysis)
number_of_rows_before_neg_qty_removal = df.shape[0]
df = df[df['Quantity'] >= 0]
number_of_rows_after_neg_qty_removal = df.shape[0]
print(f"\nNumber of rows before removing negative quantities: {number_of_rows_before_neg_qty_removal}")
print(f"Number of rows after removing negative quantities: {number_of_rows_after_neg_qty_removal}")
print(f"Rows removed due to negative quantities: {number_of_rows_before_neg_qty_removal - number_of_rows_after_neg_qty_removal}")

# 3. Handle Negative Prices (likely data entry errors)
number_of_rows_before_neg_price_removal = df.shape[0]
df = df[df['Price'] >= 0]
number_of_rows_after_neg_price_removal = df.shape[0]
print(f"\nNumber of rows before removing negative prices: {number_of_rows_before_neg_price_removal}")
print(f"Number of rows after removing negative prices: {number_of_rows_after_neg_price_removal}")
print(f"Rows removed due to negative prices: {number_of_rows_before_neg_price_removal - number_of_rows_after_neg_price_removal}")

# 4. Remove Duplicate Rows
number_of_rows_before_duplicates_removal = df.shape[0]
df.drop_duplicates(inplace=True)
number_of_rows_after_duplicates_removal = df.shape[0]
print(f"\nNumber of rows before removing duplicates: {number_of_rows_before_duplicates_removal}")
print(f"Number of rows after removing duplicates: {number_of_rows_after_duplicates_removal}")
print(f"Rows removed due to duplicates: {number_of_rows_before_duplicates_removal - number_of_rows_after_duplicates_removal}")

# 5. Create 'TotalPrice' Column
df['Total Price'] = df['Quantity'] * df['Price']
print("\n'Total Price' column created.")

print("\nData Info (after cleaning):")
df.info()

print("\nDescriptive Statistics (after cleaning):")
print(df.describe())

print("\nCleaned Data Head with 'Total Price':")
print(df.head())


# --- Phase 3: Exploratory Data Analysis (EDA) & Visualization ---

print("\n--- Phase 3: Exploratory Data Analysis (EDA) & Visualization ---")

# 1. Sales Trends Over Time
# Aggregate total price by InvoiceDate (daily sum)
df_daily_sales = df.groupby('InvoiceDate')['Total Price'].sum().reset_index()

plt.figure(figsize=(14, 7))
plt.plot(df_daily_sales['InvoiceDate'], df_daily_sales['Total Price'], marker='o', linestyle='-', markersize=2)
plt.title('Daily Total Sales Over Time', fontsize=16)
plt.xlabel('Invoice Date', fontsize=12)
plt.ylabel('Total Sales (£)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# Sales by Day of Week
df['DayOfWeek'] = df['InvoiceDate'].dt.day_name()
sales_by_day = df.groupby('DayOfWeek')['Total Price'].sum().reindex([
    'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'
]) # Reorder days for consistent plotting

plt.figure(figsize=(10, 6))
sns.barplot(x=sales_by_day.index, y=sales_by_day.values, palette='viridis')
plt.title('Total Sales by Day of Week', fontsize=16)
plt.xlabel('Day of Week', fontsize=12)
plt.ylabel('Total Sales (£)', fontsize=12)
plt.tight_layout()
plt.show()

# Sales by Hour of Day (extract hour from datetime)
df['HourOfDay'] = df['InvoiceDate'].dt.hour
sales_by_hour = df.groupby('HourOfDay')['Total Price'].sum()

plt.figure(figsize=(10, 6))
sns.lineplot(x=sales_by_hour.index, y=sales_by_hour.values, marker='o')
plt.title('Total Sales by Hour of Day', fontsize=16)
plt.xlabel('Hour of Day', fontsize=12)
plt.ylabel('Total Sales (£)', fontsize=12)
plt.xticks(range(0, 24)) # Ensure all hours are shown
plt.grid(True, linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()


# 2. Top N Products by Sales and Quantity
top_n = 10
top_products_by_quantity = df.groupby('Description')['Quantity'].sum().nlargest(top_n)
top_products_by_sales = df.groupby('Description')['Total Price'].sum().nlargest(top_n) # Corrected to use Total Price

print(f"\nTop {top_n} Products by Quantity Sold:")
print(top_products_by_quantity)

print(f"\nTop {top_n} Products by Total Sales Revenue:")
print(top_products_by_sales)

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.barplot(x=top_products_by_quantity.values, y=top_products_by_quantity.index, palette='Blues_d', ax=axes[0])
axes[0].set_title(f'Top {top_n} Products by Quantity Sold', fontsize=16)
axes[0].set_xlabel('Total Quantity Sold', fontsize=12)
axes[0].set_ylabel('Product Description', fontsize=12)

sns.barplot(x=top_products_by_sales.values, y=top_products_by_sales.index, palette='Reds_d', ax=axes[1])
axes[1].set_title(f'Top {top_n} Products by Total Sales Revenue', fontsize=16)
axes[1].set_xlabel('Total Sales Revenue (£)', fontsize=12)
axes[1].set_ylabel('Product Description', fontsize=12)

plt.tight_layout()
plt.show()


# 3. Sales by Country
sales_by_country = df.groupby('Country')['Total Price'].sum().sort_values(ascending=False)
print("\nSales by Country (Full List):")
print(sales_by_country)

top_10_countries = sales_by_country.nlargest(10)
print(f"\nTop {top_10_countries.shape[0]} Countries by Sales:")
print(top_10_countries)

plt.figure(figsize=(12, 7))
sns.barplot(x=top_10_countries.index, y=top_10_countries.values, palette='plasma')
plt.title(f'Top {top_10_countries.shape[0]} Countries by Sales', fontsize=16)
plt.xlabel('Country', fontsize=12)
plt.ylabel('Total Sales (£)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# 4. Distribution of Quantity and Price
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.histplot(df['Quantity'], bins=50, kde=True, color='skyblue', edgecolor='black', ax=axes[0])
axes[0].set_title('Distribution of Quantity per Item', fontsize=16)
axes[0].set_xlabel('Quantity', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
# Limit x-axis for better visualization due to outliers
axes[0].set_xlim(0, df['Quantity'].quantile(0.99)) # Show 99% of data

sns.histplot(df['Price'], bins=50, kde=True, color='lightcoral', edgecolor='black', ax=axes[1])
axes[1].set_title('Distribution of Price per Item', fontsize=16)
axes[1].set_xlabel('Price (£)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)
# Limit x-axis for better visualization due to outliers
axes[1].set_xlim(0, df['Price'].quantile(0.99)) # Show 99% of data

plt.tight_layout()
plt.show()


# 5. Average Order Value
# Calculate total price for each unique invoice
total_price_by_invoice = df.groupby('Invoice')['Total Price'].sum()
print("\nTotal Price for Each Unique Invoice (first 10):")
print(total_price_by_invoice.head(10))

average_order_value = total_price_by_invoice.mean()
print(f"\nAverage Order Value: £{average_order_value:.2f}")

plt.figure(figsize=(10, 6))
sns.histplot(total_price_by_invoice, bins=50, kde=True, color='purple', edgecolor='black')
plt.title('Distribution of Total Order Value per Invoice', fontsize=16)
plt.xlabel('Total Order Value (£)', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.xlim(0, total_price_by_invoice.quantile(0.99)) # Limit x-axis for better visualization
plt.tight_layout()
plt.show()


# --- Phase 4: Machine Learning (Customer Segmentation) ---

print("\n--- Phase 4: Machine Learning (Customer Segmentation) ---")

# 1. Calculate RFM Values
snapshot_date = df['InvoiceDate'].max() + timedelta(days=1)

# Recency: Days since last purchase
recency_df = df.groupby('Customer ID')['InvoiceDate'].max().reset_index()
recency_df.columns = ['Customer ID', 'LastPurchaseDate']
recency_df['Recency'] = (snapshot_date - recency_df['LastPurchaseDate']).dt.days

# Frequency: Number of unique invoices
frequency_df = df.groupby('Customer ID')['Invoice'].nunique().reset_index()
frequency_df.columns = ['Customer ID', 'Frequency']

# Monetary: Total spend
monetary_df = df.groupby('Customer ID')['Total Price'].sum().reset_index()
monetary_df.columns = ['Customer ID', 'Monetary']

# Merge all RFM features
RFM = recency_df.merge(frequency_df, on='Customer ID', how='inner')
RFM = RFM.merge(monetary_df, on='Customer ID', how='inner')

print("\nRFM DataFrame Head (before transformation/scaling):")
print(RFM.head())

# 2. Handle Skewness (Log Transformation)
# Apply log1p (log(1+x)) to handle potential zero values gracefully
RFM_log_transformed = RFM[['Recency', 'Frequency', 'Monetary']].apply(lambda x: np.log1p(x))
RFM_log_transformed.columns = ['Recency_log', 'Frequency_log', 'Monetary_log']

print("\nRFM DataFrame Head (after log transformation):")
print(RFM_log_transformed.head())

# Plot histograms of transformed RFM features
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

sns.histplot(RFM_log_transformed['Recency_log'], bins=30, kde=True, color='skyblue', edgecolor='black', ax=axes[0])
axes[0].set_title('Distribution of Log(Recency)', fontsize=16)
axes[0].set_xlabel('Log(Recency)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)

sns.histplot(RFM_log_transformed['Frequency_log'], bins=30, kde=True, color='lightcoral', edgecolor='black', ax=axes[1])
axes[1].set_title('Distribution of Log(Frequency)', fontsize=16)
axes[1].set_xlabel('Log(Frequency)', fontsize=12)
axes[1].set_ylabel('Frequency', fontsize=12)

sns.histplot(RFM_log_transformed['Monetary_log'], bins=30, kde=True, color='lightgreen', edgecolor='black', ax=axes[2])
axes[2].set_title('Distribution of Log(Monetary)', fontsize=16)
axes[2].set_xlabel('Log(Monetary)', fontsize=12)
axes[2].set_ylabel('Frequency', fontsize=12)

plt.tight_layout()
plt.show()

# 3. Feature Scaling
scaler = StandardScaler()
RFM_scaled = scaler.fit_transform(RFM_log_transformed)
RFM_scaled_df = pd.DataFrame(RFM_scaled, columns=['Recency_scaled', 'Frequency_scaled', 'Monetary_scaled'], index=RFM.index)

print("\nRFM DataFrame Head (after scaling):")
print(RFM_scaled_df.head())


# 4. Determine Optimal Number of Clusters (Elbow Method)
wcss = [] # Within-Cluster Sum of Squares
for i in range(1, 11): # Test k from 1 to 10
    kmeans = KMeans(n_clusters=i, init='k-means++', max_iter=300, n_init=10, random_state=42)
    kmeans.fit(RFM_scaled_df)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), wcss, marker='o', linestyle='--')
plt.title('Elbow Method for Optimal K', fontsize=16)
plt.xlabel('Number of Clusters (K)', fontsize=12)
plt.ylabel('Within-Cluster Sum of Squares (WCSS)', fontsize=12)
plt.grid(True, linestyle='--', alpha=0.7)
plt.show()

# Based on the elbow plot, k=4 seems to be a good choice where the curve starts to flatten.
optimal_k = 4
print(f"\nOptimal number of clusters chosen (based on Elbow Method): {optimal_k}")

# 5. Apply K-Means Clustering
kmeans = KMeans(n_clusters=optimal_k, init='k-means++', max_iter=300, n_init=10, random_state=42)
RFM['Cluster'] = kmeans.fit_predict(RFM_scaled_df)

print(f"\nRFM DataFrame Head with Cluster Labels (K={optimal_k}):")
print(RFM.head())

print("\nValue counts for each cluster:")
print(RFM['Cluster'].value_counts())

# 6. Analyze Cluster Characteristics (using original RFM values for interpretability)
cluster_analysis = RFM.groupby('Cluster')[['Recency', 'Frequency', 'Monetary']].mean().reset_index()
print("\nMean RFM Values for Each Cluster (Original Values for Interpretation):")
print(cluster_analysis)

# Sort clusters for easier comparison if needed (e.g., by Monetary value)
cluster_analysis = cluster_analysis.sort_values(by='Monetary', ascending=False).reset_index(drop=True)
print("\nMean RFM Values for Each Cluster (Sorted by Monetary Value):")
print(cluster_analysis)

# Visualizing cluster characteristics
melted_rfm = pd.melt(RFM, id_vars=['Customer ID', 'Cluster'], value_vars=['Recency', 'Frequency', 'Monetary'],
                    var_name='Metric', value_name='Value')

# Adjusting Value limits for better visualization due to high monetary values for some clusters
# Use log-transformed values for visualization if the original values are too skewed for clear plots
melted_rfm_log = pd.melt(RFM_log_transformed.assign(Cluster=RFM['Cluster'], **{'Customer ID': RFM['Customer ID']}),
                        id_vars=['Customer ID', 'Cluster'], value_vars=['Recency_log', 'Frequency_log', 'Monetary_log'],
                        var_name='Metric', value_name='Value')

plt.figure(figsize=(15, 7))
sns.boxplot(x='Metric', y='Value', hue='Cluster', data=melted_rfm_log, palette='coolwarm')
plt.title('RFM Metrics Distribution by Cluster (Log Transformed)', fontsize=16)
plt.xlabel('RFM Metric', fontsize=12)
plt.ylabel('Log Transformed Value', fontsize=12)
plt.legend(title='Cluster', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.tight_layout()
plt.show()

# --- Phase 5: Interpretation and Reporting ---

print("\n--- Phase 5: Interpretation and Reporting ---")

# 1. Summarize Key EDA Insights
eda_summary = """
Key Exploratory Data Analysis (EDA) Insights:
------------------------------------------
1.  Overall Sales Trend: The daily sales plot shows significant fluctuations over time, with clear periods of higher and lower activity. Further time series decomposition would be needed to precisely identify underlying trend and seasonality. Sales are generally higher on Tuesdays and Thursdays, and peak around midday (10 AM to 3 PM).
2.  Geographical Sales Distribution: The United Kingdom is by far the dominant market, accounting for the vast majority of total sales revenue. EIRE, Netherlands, Germany, and France follow, but their contributions are substantially smaller, indicating a highly concentrated market.
3.  Top Products: Products like 'WHITE HANGING HEART T-LIGHT HOLDER', 'WORLD WAR 2 GLIDERS ASSTD DESIGNS', and 'BROCADE RING PURSE' are consistently top performers both in terms of quantity sold and total revenue generated. These are likely popular, potentially high-volume items.
4.  Distribution of Quantity and Price: Both 'Quantity' and 'Price' distributions are highly skewed to the lower end, meaning most transactions involve small quantities of relatively low-priced items. There's a long tail indicating a few very large or expensive purchases.
5.  Average Order Value: The distribution of total order values per invoice also shows skewness, with most transactions being of lower monetary value, and a smaller number of high-value orders. The average order value is around £X (refer to your specific average_order_value calculation).
"""
print(eda_summary)
print(f"Average Order Value: £{average_order_value:.2f}") # Dynamic inclusion

# 2. Detailed Customer Segment Characterization
customer_segments_description = """
Customer Segment Characterization (Based on RFM Analysis with 4 Clusters):
----------------------------------------------------------------------
(Note: Cluster IDs may vary based on K-Means initialization, but characteristics remain consistent)

Here, we'll map the actual cluster IDs from the 'cluster_analysis' DataFrame to descriptive names.
Let's assume the sorting by Monetary value places the 'Champions' first.

-   **Cluster {0_id}: "Champions / Most Valuable Customers"**
    -   **Recency (Mean):** ~{0_recency:.2f} days (Very Low) - Purchased very recently.
    -   **Frequency (Mean):** ~{0_frequency:.0f} (Very High) - Extremely frequent buyers.
    -   **Monetary (Mean):** ~£{0_monetary:.2f} (Very High) - Top revenue contributors.
    -   **Description:** These are the most loyal, active, and highest-spending customers. They are critical to the business's success.

-   **Cluster {1_id}: "At-Risk Loyal Customers"**
    -   **Recency (Mean):** ~{1_recency:.2f} days (Medium to High) - Noticeable gap since last purchase.
    -   **Frequency (Mean):** ~{1_frequency:.0f} (High) - Previously very active.
    -   **Monetary (Mean):** ~£{1_monetary:.2f} (High) - Significant past contributions.
    -   **Description:** Customers who were highly valuable but are showing signs of disengagement. They are at risk of churning if not re-engaged soon.

-   **Cluster {2_id}: "Promising / Active Customers"**
    -   **Recency (Mean):** ~{2_recency:.2f} days (Low) - Relatively recent engagement.
    -   **Frequency (Mean):** ~{2_frequency:.0f} (Moderate) - Consistent but not extremely frequent.
    -   **Monetary (Mean):** ~£{2_monetary:.2f} (Moderate) - Contribute reasonably well.
    -   **Description:** Active customers with potential to grow into champions with proper nurturing. They might be newer or buy steadily.

-   **Cluster {3_id}: "Churned / Lost Customers"**
    -   **Recency (Mean):** ~{3_recency:.2f} days (Very High) - Long time since last purchase.
    -   **Frequency (Mean):** ~{3_frequency:.0f} (Very Low) - Infrequent buyers.
    -   **Monetary (Mean):** ~£{3_monetary:.2f} (Very Low) - Minimal past spend.
    -   **Description:** Largely inactive customers who have likely churned. They represent low return on re-engagement efforts.
""".format(
    # Dynamically map the cluster IDs and their mean values based on sorted 'cluster_analysis'
    # Assuming cluster_analysis is sorted by Monetary value descending as done above
    **{f"{i}_id": int(cluster_analysis.iloc[i]['Cluster']) for i in range(optimal_k)},
    **{f"{i}_recency": cluster_analysis.iloc[i]['Recency'] for i in range(optimal_k)},
    **{f"{i}_frequency": cluster_analysis.iloc[i]['Frequency'] for i in range(optimal_k)},
    **{f"{i}_monetary": cluster_analysis.iloc[i]['Monetary'] for i in range(optimal_k)}
)
print(customer_segments_description)


# 3. Actionable Business Recommendations
business_recommendations = """
Actionable Business Recommendations:
----------------------------------
Based on our EDA and customer segmentation, here are strategic recommendations:

1.  **For "Champions" (Cluster with highest Monetary):**
    * **Retention & Reward:** Implement exclusive loyalty programs, early access to new products, VIP customer service, and personalized 'thank you' gestures. Focus on appreciation over discounts to maintain perceived value.
    * **Upselling/Cross-selling:** Offer premium product lines or complementary items based on their past purchases.
    * **Advocacy Programs:** Encourage referrals and testimonials, as they are likely to be strong brand advocates.

2.  **For "Promising / Active Customers" (Cluster with moderate Recency, Frequency, Monetary):**
    * **Nurturing & Growth:** Provide personalized product recommendations and curated content based on their buying patterns.
    * **Engagement:** Send targeted email campaigns with product updates, special offers, or relevant information to encourage repeat purchases.
    * **Seamless Experience:** Ensure consistent, quick, and positive customer service experiences to build stronger loyalty.

3.  **For "At-Risk Loyal Customers" (Cluster with higher Recency, but high past F & M):**
    * **Re-engagement Campaigns:** Implement personalized win-back strategies, such as "We miss you" emails with enticing, time-limited discounts or free shipping.
    * **Feedback Collection:** Proactively reach out to understand reasons for reduced activity.
    * **Highlight Value:** Remind them of the benefits and value they received from past purchases.

4.  **For "Churned / Lost Customers" (Cluster with very high Recency, low F & M):**
    * **Selective Re-acquisition:** For a very small, carefully identified sub-segment, extremely attractive offers might be tested, but generally, resource allocation should be minimal as ROI is low.
    * **Churn Prevention Analysis:** Analyze their characteristics to refine strategies for "At-Risk" customers and prevent future churn.
    * **Surveys:** Consider short, anonymous surveys to gather insights into reasons for complete churn.

5.  **General Business Recommendations (from EDA):**
    * **Geographical Strategy:** Continue to prioritize the UK market as the primary revenue driver. Explore localized marketing and logistics investments in secondary markets like EIRE, Netherlands, Germany, and France.
    * **Product Focus:** Ensure consistent stock and promotion of top-selling products. Explore bundling popular items to increase average order value.
    * **Inventory & Pricing:** The skewed distributions of quantity and price highlight the importance of robust inventory management for high-volume, low-cost items and careful pricing strategies for higher-value, lower-volume goods.
    * **Time-based Optimization:** Leverage insights from daily/hourly sales trends for staffing, marketing campaign timing, and product availability.
"""
print(business_recommendations)

# 4. Limitations and Future Work
limitations_future_work = """
Limitations and Future Work:
--------------------------
1.  **Data Scope:** This analysis primarily uses transactional data. Integrating customer demographic information (e.g., age, gender, precise location) or website Browse behavior would enable even richer segmentation and more personalized strategies.
2.  **Time Period:** The dataset covers a specific time frame. Incorporating a longer historical period would provide deeper insights into long-term customer behavior trends and seasonality.
3.  **Customer ID Gaps:** The exclusion of transactions without a Customer ID means we only analyzed a subset of all sales. Future work could investigate these anonymous transactions for overall revenue insights.
4.  **Clustering Algorithm:** While K-Means is effective, exploring other clustering algorithms (e.g., DBSCAN for density-based clusters or Hierarchical Clustering) might reveal alternative or more nuanced segmentations.
5.  **Predictive Modeling:** This project focuses on descriptive and segmentation analysis. Future work could involve building predictive models such as:
    * **Customer Lifetime Value (CLTV) Prediction:** To forecast the future revenue contribution of each customer.
    * **Churn Prediction:** To proactively identify customers at high risk of churning before they become 'At-Risk'.
    * **Recommendation Systems:** To provide highly personalized product suggestions, enhancing customer experience and driving sales.
    * **Time Series Forecasting:** To accurately predict future sales volumes or demand for specific products.
"""
print(limitations_future_work)
