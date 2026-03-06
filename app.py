import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Sales Dashboard", layout="wide")

st.title("Sales Data Analysis Dashboard")

# Load data
df = pd.read_csv("data/sales.csv")
df.columns = df.columns.str.strip()

filtered_df = df.copy()
col1, col2, col3 = st.columns(3)

with col1:
    st.metric("Total Sales", round(filtered_df["Sales"].sum(), 2))

with col2:
    st.metric("Total Orders", filtered_df["Order ID"].nunique())

with col3:
    st.metric("Total Customers", filtered_df["Customer ID"].nunique())

st.download_button(
    "Download filtered data",
    filtered_df.to_csv(index=False),
    file_name="filtered_sales.csv",
    mime="text/csv"
)






# Convert date column
df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)

# ---------------- Sidebar Filters ----------------

st.sidebar.header("Filters")

region = st.sidebar.multiselect(
    "Select Region",
    options=df["Region"].unique(),
    default=df["Region"].unique()
)

category = st.sidebar.multiselect(
    "Select Category",
    options=df["Category"].unique(),
    default=df["Category"].unique()
)

min_date = df["Order Date"].min()
max_date = df["Order Date"].max()

date_range = st.sidebar.date_input(
    "Select Date Range",
    [min_date, max_date]
)

# ---------------- Filtered Data ----------------

filtered_df = df[
    (df["Region"].isin(region)) &
    (df["Category"].isin(category)) &
    (df["Order Date"] >= pd.to_datetime(date_range[0])) &
    (df["Order Date"] <= pd.to_datetime(date_range[1]))
]

# ---------------- KPIs ----------------

total_sales = filtered_df["Sales"].sum()
total_profit = filtered_df["Sales"].sum() 

total_orders = filtered_df.shape[0]
avg_order_value = total_sales / total_orders if total_orders > 0 else 0

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Sales", f"{total_sales:,.2f}")
col2.metric("Total Profit", f"{total_profit:,.2f}")
col3.metric("Total Orders", total_orders)
col4.metric("Avg Order Value", f"{avg_order_value:,.2f}")




st.markdown("---")

# ---------------- Monthly Sales ----------------

filtered_df["Month"] = filtered_df["Order Date"].dt.to_period("M").astype(str)

monthly_sales = (
    filtered_df.groupby("Month")["Sales"]
    .sum()
    .reset_index()
)

fig1 = px.line(
    monthly_sales,
    x="Month",
    y="Sales",
    title="Monthly Sales Trend"
)

st.plotly_chart(fig1, use_container_width=True)

# ------------ Best Selling Month ------------

best_month = monthly_sales.loc[monthly_sales["Sales"].idxmax()]

st.success(
    f"Best Selling Month: {best_month['Month']} with Sales {best_month['Sales']:,.2f}"
)



# ------------ Sales Forecast ------------

from prophet import Prophet

st.subheader("Sales Forecast")

forecast_df = filtered_df[["Order Date", "Sales"]]
forecast_df = forecast_df.rename(columns={"Order Date": "ds", "Sales": "y"})

model = Prophet()
model.fit(forecast_df)

future = model.make_future_dataframe(periods=30)
forecast = model.predict(future)

fig_forecast = px.line(forecast, x="ds", y="yhat", title="Sales Forecast (Next 30 Days)")

st.plotly_chart(fig_forecast, use_container_width=True)


# ---------------- Sales by Category ----------------

profit_by_category = (
    filtered_df.groupby("Category")["Sales"]
    .sum()
    .reset_index()
)

fig4 = px.bar(
    profit_by_category,
    x="Category",
    y="Sales",
    title="Sales by Category"
)



st.plotly_chart(fig4, use_container_width=True, key="profit_by_category")





# ---------------- Region wise sales ----------------

region_sales = (
    filtered_df.groupby("Region")["Sales"]
    .sum()
    .reset_index()
)

fig3 = px.pie(
    region_sales,
    names="Region",
    values="Sales",
    title="Sales by Region"
)

st.plotly_chart(fig3, use_container_width=True)

# ------------ Category vs Region Heatmap ------------

st.subheader("Category vs Region Sales Heatmap")

heatmap_data = (
    filtered_df.pivot_table(
        values="Sales",
        index="Category",
        columns="Region",
        aggfunc="sum"
    )
)

fig6 = px.imshow(
    heatmap_data,
    text_auto=True,
    aspect="auto",
    title="Sales Distribution"
)

st.plotly_chart(fig6, use_container_width=True)


# ------------ Top 10 Products by Sales ------------

st.subheader("Top 10 Products by Sales")

top_products = (
    filtered_df.groupby("Product Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
)

st.bar_chart(top_products)

# ------------ Top 10 Customers by Sales ------------

st.subheader("Top 10 Customers by Sales")

top_customers = (
    filtered_df.groupby("Customer Name")["Sales"]
    .sum()
    .sort_values(ascending=False)
    .head(10)
    .reset_index()
)

fig5 = px.bar(
    top_customers,
    x="Customer Name",
    y="Sales",
    title="Top Customers by Sales"
)

st.plotly_chart(fig5, use_container_width=True)



# ---------------- Data Table ----------------

st.subheader("Filtered Sales Data")
st.dataframe(filtered_df)



# ------------ Interactive Search ------------

st.markdown("### Search in Data")

search = st.text_input("Search product or customer")

if search:
    search_df = filtered_df[
        filtered_df.apply(lambda row: row.astype(str).str.contains(search, case=False).any(), axis=1)
    ]
    st.dataframe(search_df)





st.markdown("---")
st.caption("Sales Data Analysis Dashboard | Built with Python, Pandas, Plotly & Streamlit")
