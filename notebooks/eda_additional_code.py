# ============================================================================
# QUESTION 1: Orders by Month & Year Timeline (Bar Chart)
# ============================================================================

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Ensure dates are datetime (should already be from earlier cells)
orders_master['order_purchase_timestamp'] = pd.to_datetime(orders_master['order_purchase_timestamp'])

# Extract year-month
orders_master['year_month'] = orders_master['order_purchase_timestamp'].dt.to_period('M')

# Count orders by month
orders_by_month = orders_master.groupby('year_month').size().reset_index(name='order_count')

# Convert back to timestamp for plotting
orders_by_month['year_month_ts'] = orders_by_month['year_month'].dt.to_timestamp()

# Create timeline bar chart
plt.figure(figsize=(16, 6))
plt.bar(orders_by_month['year_month_ts'], orders_by_month['order_count'], width=20, color='steelblue', edgecolor='navy', alpha=0.7)
plt.xlabel('Month-Year', fontsize=12)
plt.ylabel('Number of Orders', fontsize=12)
plt.title('Orders Timeline: Monthly Order Volume', fontsize=14, fontweight='bold')
plt.xticks(rotation=45, ha='right')
plt.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.show()

# Print summary stats
print("\n===== MONTHLY ORDER STATISTICS =====")
print(f"Total orders: {orders_by_month['order_count'].sum()}")
print(f"Average orders/month: {orders_by_month['order_count'].mean():.0f}")
print(f"Peak month: {orders_by_month.loc[orders_by_month['order_count'].idxmax(), 'year_month']} ({orders_by_month['order_count'].max()} orders)")
print(f"Lowest month: {orders_by_month.loc[orders_by_month['order_count'].idxmin(), 'year_month']} ({orders_by_month['order_count'].min()} orders)")
print("\nMonth-by-Month Breakdown:")
print(orders_by_month[['year_month', 'order_count']].to_string(index=False))


# ============================================================================
# QUESTION 2: Seller Delay vs Total Lateness Correlation
# Attribution Analysis: Is lateness due to Seller or Freight?
# ============================================================================

import numpy as np

# Ensure all date columns are datetime
orders_master['order_purchase_timestamp'] = pd.to_datetime(orders_master['order_purchase_timestamp'])
orders_master['order_delivered_carrier_date'] = pd.to_datetime(orders_master['order_delivered_carrier_date'])
orders_master['order_delivered_customer_date'] = pd.to_datetime(orders_master['order_delivered_customer_date'])
orders_master['order_estimated_delivery_date'] = pd.to_datetime(orders_master['order_estimated_delivery_date'])

# Calculate delays in days
# Seller delay: Time from order placed to handoff to carrier
orders_master['seller_delay_days'] = (
    orders_master['order_delivered_carrier_date'] -
    orders_master['order_purchase_timestamp']
).dt.days

# Logistics delay: Time from carrier pickup to customer delivery
orders_master['logistics_delay_days'] = (
    orders_master['order_delivered_customer_date'] -
    orders_master['order_delivered_carrier_date']
).dt.days

# Total lateness: How many days late relative to promised date
orders_master['total_lateness_days'] = (
    orders_master['order_delivered_customer_date'] -
    orders_master['order_estimated_delivery_date']
).dt.days

# Filter to delivered orders only (to avoid NaT values)
delivery_analysis = orders_master[
    orders_master['order_status'] == 'delivered'
].copy()

# Remove rows with any NaT values in key columns
delivery_analysis = delivery_analysis.dropna(subset=[
    'seller_delay_days', 'logistics_delay_days', 'total_lateness_days'
])

print("\n===== DELAY BREAKDOWN ANALYSIS =====")
print(f"\nTotal delivered orders analyzed: {len(delivery_analysis)}")

print("\n--- Seller Delay (Order Placed to Carrier Pickup) ---")
print(delivery_analysis['seller_delay_days'].describe())
print(f"Median: {delivery_analysis['seller_delay_days'].median():.1f} days")
print(f"95th percentile: {delivery_analysis['seller_delay_days'].quantile(0.95):.1f} days")

print("\n--- Logistics Delay (Carrier Pickup to Customer Delivery) ---")
print(delivery_analysis['logistics_delay_days'].describe())
print(f"Median: {delivery_analysis['logistics_delay_days'].median():.1f} days")
print(f"95th percentile: {delivery_analysis['logistics_delay_days'].quantile(0.95):.1f} days")

print("\n--- Total Lateness (vs Promised Date) ---")
print(delivery_analysis['total_lateness_days'].describe())
print(f"Median: {delivery_analysis['total_lateness_days'].median():.1f} days")
print(f"% of orders that were late: {(delivery_analysis['total_lateness_days'] > 0).mean() * 100:.2f}%")
print(f"% of orders that were early: {(delivery_analysis['total_lateness_days'] < 0).mean() * 100:.2f}%")

# ===== CORRELATION ANALYSIS =====
print("\n\n===== CORRELATION: WHO'S RESPONSIBLE FOR LATENESS? =====")

correlation_matrix = delivery_analysis[[
    'seller_delay_days',
    'logistics_delay_days',
    'total_lateness_days',
    'distance_km'
]].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

# Focus on the key insight
seller_lateness_corr = delivery_analysis['seller_delay_days'].corr(delivery_analysis['total_lateness_days'])
logistics_lateness_corr = delivery_analysis['logistics_delay_days'].corr(delivery_analysis['total_lateness_days'])

print(f"\n📊 KEY FINDING:")
print(f"   Seller Delay ↔ Total Lateness:     {seller_lateness_corr:.4f}")
print(f"   Logistics Delay ↔ Total Lateness: {logistics_lateness_corr:.4f}")

if abs(seller_lateness_corr) > abs(logistics_lateness_corr):
    print(f"\n🎯 CONCLUSION: Seller delays are MORE correlated with total lateness")
    print(f"   → Seller handling is the primary driver of lateness issues")
else:
    print(f"\n🎯 CONCLUSION: Logistics delays are MORE correlated with total lateness")
    print(f"   → Freight/carrier is the primary driver of lateness issues")

# ===== VISUALIZATION 1: Distribution Comparison =====
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].hist(delivery_analysis['seller_delay_days'], bins=50, color='coral', edgecolor='black', alpha=0.7)
axes[0].set_xlabel('Days', fontsize=11)
axes[0].set_ylabel('Frequency', fontsize=11)
axes[0].set_title(f'Seller Delay Distribution\n(Order → Carrier Pickup)', fontsize=12, fontweight='bold')
axes[0].axvline(delivery_analysis['seller_delay_days'].median(), color='red', linestyle='--', linewidth=2, label=f"Median: {delivery_analysis['seller_delay_days'].median():.1f}d")
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].hist(delivery_analysis['logistics_delay_days'], bins=50, color='skyblue', edgecolor='black', alpha=0.7)
axes[1].set_xlabel('Days', fontsize=11)
axes[1].set_ylabel('Frequency', fontsize=11)
axes[1].set_title(f'Logistics Delay Distribution\n(Carrier Pickup → Customer Delivery)', fontsize=12, fontweight='bold')
axes[1].axvline(delivery_analysis['logistics_delay_days'].median(), color='red', linestyle='--', linewidth=2, label=f"Median: {delivery_analysis['logistics_delay_days'].median():.1f}d")
axes[1].legend()
axes[1].grid(alpha=0.3)

axes[2].hist(delivery_analysis['total_lateness_days'], bins=50, color='lightgreen', edgecolor='black', alpha=0.7)
axes[2].set_xlabel('Days (+ = Late, - = Early)', fontsize=11)
axes[2].set_ylabel('Frequency', fontsize=11)
axes[2].set_title(f'Total Lateness Distribution\n(vs Promised Delivery Date)', fontsize=12, fontweight='bold')
axes[2].axvline(0, color='orange', linestyle='-', linewidth=2, label='On-Time Mark')
axes[2].axvline(delivery_analysis['total_lateness_days'].median(), color='red', linestyle='--', linewidth=2, label=f"Median: {delivery_analysis['total_lateness_days'].median():.1f}d")
axes[2].legend()
axes[2].grid(alpha=0.3)

plt.tight_layout()
plt.show()

# ===== VISUALIZATION 2: Scatter Plot - Seller vs Logistics Delay =====
fig, ax = plt.subplots(figsize=(12, 8))

scatter = ax.scatter(
    delivery_analysis['seller_delay_days'],
    delivery_analysis['logistics_delay_days'],
    c=delivery_analysis['total_lateness_days'],
    cmap='RdYlGn_r',  # Red = late, Yellow = on-time, Green = early
    alpha=0.5,
    s=30,
    edgecolors='black',
    linewidth=0.5
)

ax.set_xlabel('Seller Delay (days)', fontsize=12, fontweight='bold')
ax.set_ylabel('Logistics Delay (days)', fontsize=12, fontweight='bold')
ax.set_title('Seller vs Logistics Delay Attribution\n(Color = Total Lateness: Red=Late, Green=Early)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3)

cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Total Lateness (days)', fontsize=11)

plt.tight_layout()
plt.show()

# ===== VISUALIZATION 3: Average Lateness by Seller Delay Bucket =====
# Bucket seller delays to see impact
delivery_analysis['seller_delay_bucket'] = pd.cut(
    delivery_analysis['seller_delay_days'],
    bins=[0, 5, 10, 15, 20, 1000],
    labels=['0-5 days', '6-10 days', '11-15 days', '16-20 days', '20+ days']
)

lateness_by_seller_delay = delivery_analysis.groupby('seller_delay_bucket')['total_lateness_days'].agg([
    'mean', 'median', 'std', 'count'
]).reset_index()

fig, ax = plt.subplots(figsize=(12, 6))

x_pos = np.arange(len(lateness_by_seller_delay))
bars = ax.bar(x_pos, lateness_by_seller_delay['mean'],
              color='steelblue', alpha=0.7, edgecolor='navy', linewidth=1.5)

# Add count labels on bars
for i, (bar, count) in enumerate(zip(bars, lateness_by_seller_delay['count'])):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'n={int(count)}',
            ha='center', va='bottom', fontsize=9)

ax.set_xlabel('Seller Delay Duration', fontsize=12, fontweight='bold')
ax.set_ylabel('Average Total Lateness (days)', fontsize=12, fontweight='bold')
ax.set_title('Impact of Seller Delay on Total Order Lateness', fontsize=13, fontweight='bold')
ax.set_xticks(x_pos)
ax.set_xticklabels(lateness_by_seller_delay['seller_delay_bucket'])
ax.axhline(0, color='red', linestyle='--', linewidth=2, alpha=0.5, label='On-Time Threshold')
ax.grid(axis='y', alpha=0.3)
ax.legend()

plt.tight_layout()
plt.show()

print("\n📈 Data for the bar chart:")
print(lateness_by_seller_delay.to_string(index=False))

# ===== SUMMARY TABLE =====
print("\n\n===== SUMMARY: DELAY ATTRIBUTION =====")
print("\nAverage Delays:")
print(f"  Seller Delay (Order → Carrier):        {delivery_analysis['seller_delay_days'].mean():.1f} days")
print(f"  Logistics Delay (Carrier → Customer):  {delivery_analysis['logistics_delay_days'].mean():.1f} days")
print(f"  Total Delay (if order is late):        {delivery_analysis[delivery_analysis['total_lateness_days'] > 0]['total_lateness_days'].mean():.1f} days")

print("\nMedian Delays:")
print(f"  Seller Delay (Order → Carrier):        {delivery_analysis['seller_delay_days'].median():.1f} days")
print(f"  Logistics Delay (Carrier → Customer):  {delivery_analysis['logistics_delay_days'].median():.1f} days")
print(f"  Total Lateness (if order is late):     {delivery_analysis[delivery_analysis['total_lateness_days'] > 0]['total_lateness_days'].median():.1f} days")

print("\n✅ INTERPRETATION:")
print("  • High seller_lateness_corr → Sellers are slow at preparing/handing off orders")
print("  • High logistics_lateness_corr → Carriers are slow in transit")
print("  • Use this to decide: Should you optimize seller processes or negotiate with logistics partners?")
