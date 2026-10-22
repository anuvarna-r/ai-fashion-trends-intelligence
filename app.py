import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from pathlib import Path


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="FashionIQ | Fashion Intelligence",
    page_icon="👗",
    layout="wide"
)


# =====================================================
# LOAD DATA
# =====================================================

@st.cache_data
def load_data():
    df = pd.read_csv("data/fashion_monthly_sales.csv")

    df["Month"] = pd.to_datetime(df["Month"])

    # Exclude incomplete March 2025 data
    df = df[df["Month"] < "2025-03-01"].copy()

    return df


@st.cache_data
def load_model_comparison():
    extended_path = Path("models/extended_model_comparison.csv")
    original_path = Path("models/model_comparison.csv")

    if extended_path.exists():
        return pd.read_csv(extended_path), "Extended"

    if original_path.exists():
        return pd.read_csv(original_path), "Original"

    return None, None


@st.cache_data
def load_model_predictions():
    extended_path = Path("models/extended_model_predictions.csv")
    original_path = Path("models/random_forest_predictions.csv")

    if extended_path.exists():
        return pd.read_csv(extended_path)

    if original_path.exists():
        return pd.read_csv(original_path)

    return None


df = load_data()

model_results, evaluation_type = load_model_comparison()

model_predictions = load_model_predictions()


# =====================================================
# SIDEBAR
# =====================================================

st.sidebar.title("👗 FashionIQ")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Fashion Trends",
        "Demand Forecasting",
        "Model Performance"
    ]
)

st.sidebar.divider()

st.sidebar.caption(
    "AI Fashion Trends Intelligence & Forecasting System"
)

st.sidebar.caption(
    "Historical Data: January 2023 – February 2025"
)


# =====================================================
# HELPER FUNCTIONS
# =====================================================

def category_segments(data):
    return data[
        ["Category", "Sub Category"]
    ].drop_duplicates().shape[0]


def monthly_totals(data):
    return (
        data.groupby("Month", as_index=False)["Units_Sold"]
        .sum()
        .sort_values("Month")
    )


def top_segments(data, n=10):

    result = (
        data.groupby(
            ["Category", "Sub Category"],
            as_index=False
        )["Units_Sold"]
        .sum()
        .sort_values(
            "Units_Sold",
            ascending=False
        )
        .head(n)
    )

    result["Segment"] = (
        result["Category"]
        + " - "
        + result["Sub Category"]
    )

    return result


def display_monthly_chart(data, title):

    monthly = monthly_totals(data)

    fig = px.line(
        monthly,
        x="Month",
        y="Units_Sold",
        markers=True,
        title=title
    )

    fig.update_layout(
        xaxis_title="Month",
        yaxis_title="Units Sold"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


def display_top_segments(data):

    top = top_segments(data)

    fig = px.bar(
        top,
        x="Units_Sold",
        y="Segment",
        orientation="h",
        color="Units_Sold",
        color_continuous_scale="Teal",
        title="Top 10 Fashion Segments"
    )

    fig.update_layout(
        yaxis={
            "categoryorder": "total ascending"
        },
        xaxis_title="Units Sold",
        yaxis_title=""
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


# =====================================================
# DASHBOARD
# =====================================================

if page == "Dashboard":

    st.title("👗 FashionIQ")

    st.subheader(
        "Fashion Intelligence Dashboard"
    )

    st.write(
        "Explore fashion demand, category performance, "
        "and historical seasonal trends."
    )

    total_units = int(
        df["Units_Sold"].sum()
    )

    segments = category_segments(df)

    months = df["Month"].nunique()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Total Units Sold",
        f"{total_units:,}"
    )

    col2.metric(
        "Fashion Segments",
        segments
    )

    col3.metric(
        "Months Analyzed",
        months
    )

    st.divider()

    st.subheader(
        "📈 Monthly Fashion Sales"
    )

    display_monthly_chart(
        df,
        "Historical Monthly Fashion Demand"
    )

    st.subheader(
        "🏆 Best-Selling Fashion Segments"
    )

    display_top_segments(df)

    st.subheader(
        "🛍️ Sales Distribution"
    )

    category_sales = (
        df.groupby(
            "Category",
            as_index=False
        )["Units_Sold"]
        .sum()
    )

    fig = px.pie(
        category_sales,
        names="Category",
        values="Units_Sold",
        hole=0.45,
        title="Sales Distribution by Fashion Category",
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )


# =====================================================
# FASHION TRENDS
# =====================================================

elif page == "Fashion Trends":

    st.title(
        "📈 Fashion Trends Intelligence"
    )

    st.write(
        "Analyze fashion category demand, "
        "top-selling segments, and seasonal purchasing patterns."
    )

    selected_category = st.selectbox(
        "Select Fashion Category",
        ["All Categories"]
        + sorted(
            df["Category"].unique().tolist()
        )
    )

    if selected_category == "All Categories":
        filtered_df = df.copy()

    else:
        filtered_df = df[
            df["Category"] == selected_category
        ].copy()

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Units Sold",
        f"{int(filtered_df['Units_Sold'].sum()):,}"
    )

    col2.metric(
        "Fashion Segments",
        category_segments(filtered_df)
    )

    col3.metric(
        "Months Analyzed",
        filtered_df["Month"].nunique()
    )

    st.divider()

    st.subheader(
        "📊 Monthly Demand"
    )

    display_monthly_chart(
        filtered_df,
        "Monthly Fashion Demand Trends"
    )

    st.subheader(
        "🏆 Top-Selling Segments"
    )

    display_top_segments(filtered_df)

    st.subheader(
        "🥧 Category Distribution"
    )

    category_distribution = (
        filtered_df.groupby(
            "Category",
            as_index=False
        )["Units_Sold"]
        .sum()
    )

    fig = px.pie(
        category_distribution,
        names="Category",
        values="Units_Sold",
        hole=0.4,
        title="Fashion Category Distribution",
        color_discrete_sequence=px.colors.qualitative.Set2
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.subheader(
        "🌸 Seasonal Demand Analysis"
    )

    seasonal = monthly_totals(filtered_df)

    seasonal["Month_Number"] = (
        seasonal["Month"].dt.month
    )

    seasonal["Month_Name"] = (
        seasonal["Month"].dt.strftime("%B")
    )

    seasonal_average = (
        seasonal.groupby(
            ["Month_Number", "Month_Name"],
            as_index=False
        )["Units_Sold"]
        .mean()
        .sort_values("Month_Number")
    )

    fig = px.bar(
        seasonal_average,
        x="Month_Name",
        y="Units_Sold",
        color="Units_Sold",
        color_continuous_scale="Purples",
        title="Average Total Demand by Calendar Month"
    )

    fig.update_layout(
        xaxis_title="Calendar Month",
        yaxis_title="Average Units Sold"
    )

    st.plotly_chart(
        fig,
        width="stretch"
    )

    st.caption(
        "Seasonal averages use complete months "
        "from January 2023 to February 2025."
    )

    st.subheader(
        "📋 Historical Fashion Data"
    )

    st.dataframe(
        filtered_df.sort_values(
            "Month",
            ascending=False
        ),
        width="stretch"
    )


# =====================================================
# DEMAND FORECASTING
# =====================================================

elif page == "Demand Forecasting":

    st.title(
        "🔮 Fashion Demand Forecasting"
    )

    st.write(
        "Estimate future monthly fashion demand "
        "using historical seasonal patterns."
    )

    st.info(
        "The forecasting method uses sales from "
        "the same calendar month of the previous year."
    )

    category = st.selectbox(
        "Select Fashion Category",
        sorted(
            df["Category"].unique().tolist()
        )
    )

    available_subcategories = sorted(
        df.loc[
            df["Category"] == category,
            "Sub Category"
        ].unique().tolist()
    )

    subcategory = st.selectbox(
        "Select Fashion Subcategory",
        available_subcategories
    )

    selected_data = df[
        (df["Category"] == category)
        & (df["Sub Category"] == subcategory)
    ].copy()

    selected_data = (
        selected_data.groupby(
            "Month",
            as_index=False
        )["Units_Sold"]
        .sum()
        .sort_values("Month")
    )

    forecast_months = pd.date_range(
        start="2025-03-01",
        end="2026-02-01",
        freq="MS"
    )

    forecast_options = {
        date.strftime("%B %Y"): date
        for date in forecast_months
    }

    selected_label = st.selectbox(
        "Select Forecast Month",
        list(forecast_options.keys())
    )

    selected_month = forecast_options[
        selected_label
    ]

    historical_lookup = selected_data.set_index(
        "Month"
    )["Units_Sold"]

    forecast_records = []

    for month in forecast_months:

        source_month = (
            month - pd.DateOffset(years=1)
        )

        if source_month in historical_lookup.index:

            forecast_records.append({
                "Forecast Month": month,
                "Reference Month": source_month,
                "Predicted Units": int(
                    historical_lookup.loc[source_month]
                )
            })

    forecast_table = pd.DataFrame(
        forecast_records
    )

    if forecast_table.empty:

        st.error(
            "Historical data is unavailable "
            "for this forecast."
        )

    else:

        selected_prediction = forecast_table[
            forecast_table["Forecast Month"]
            == selected_month
        ]

        predicted_units = int(
            selected_prediction[
                "Predicted Units"
            ].iloc[0]
        )

        latest_actual = int(
            selected_data["Units_Sold"].iloc[-1]
        )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Forecast Month",
            selected_label
        )

        col2.metric(
            "Predicted Demand",
            f"{predicted_units:,} units"
        )

        col3.metric(
            "Latest Recorded Monthly Sales",
            f"{latest_actual:,} units"
        )

        st.divider()

        st.subheader(
            "📈 Historical Sales & 12-Month Forecast"
        )

        fig = go.Figure()

        fig.add_trace(
            go.Scatter(
                x=selected_data["Month"],
                y=selected_data["Units_Sold"],
                mode="lines+markers",
                name="Historical Sales",
                line=dict(
                    color="#1688E8",
                    width=3
                )
            )
        )

        fig.add_trace(
            go.Scatter(
                x=forecast_table["Forecast Month"],
                y=forecast_table["Predicted Units"],
                mode="lines+markers",
                name="Forecasted Demand",
                line=dict(
                    color="#A83CC5",
                    width=3,
                    dash="dash"
                )
            )
        )

        fig.add_trace(
            go.Scatter(
                x=[selected_month],
                y=[predicted_units],
                mode="markers",
                name="Selected Forecast Month",
                marker=dict(
                    color="#FF9500",
                    size=13
                )
            )
        )

        fig.update_layout(
            title=(
                f"{category} - {subcategory}: "
                "Historical Sales vs Forecast"
            ),
            xaxis_title="Month",
            yaxis_title="Units Sold"
        )

        st.plotly_chart(
            fig,
            width="stretch"
        )

        st.subheader(
            "📊 Forecast Summary"
        )

        total_forecast = int(
            forecast_table["Predicted Units"].sum()
        )

        average_forecast = int(
            forecast_table["Predicted Units"].mean()
        )

        peak_row = forecast_table.loc[
            forecast_table["Predicted Units"].idxmax()
        ]

        peak_month = peak_row[
            "Forecast Month"
        ].strftime("%B %Y")

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "12-Month Forecast Total",
            f"{total_forecast:,}"
        )

        col2.metric(
            "Average Monthly Demand",
            f"{average_forecast:,}"
        )

        col3.metric(
            "Peak Demand Month",
            peak_month
        )

        st.subheader(
            "📋 Monthly Demand Forecast"
        )

        display_table = forecast_table.copy()

        display_table["Forecast Month"] = (
            display_table["Forecast Month"]
            .dt.strftime("%B %Y")
        )

        display_table["Reference Month"] = (
            display_table["Reference Month"]
            .dt.strftime("%B %Y")
        )

        st.dataframe(
            display_table,
            width="stretch",
            hide_index=True
        )

        csv = forecast_table.to_csv(
            index=False
        ).encode("utf-8")

        st.download_button(
            "📥 Download Forecast CSV",
            data=csv,
            file_name="fashioniq_forecast.csv",
            mime="text/csv"
        )

        st.warning(
            "These forecasts cover March 2025 to February 2026 "
            "using historical seasonal patterns. "
            "They are not current market forecasts and "
            "have not been validated against actual sales "
            "for those months."
        )


# =====================================================
# MODEL PERFORMANCE
# =====================================================

elif page == "Model Performance":

    st.title(
        "🧠 Forecasting Model Performance"
    )

    st.write(
        "Compare forecasting models and "
        "evaluate their prediction accuracy."
    )

    if model_results is None:

        st.error(
            "Model comparison results were not found."
        )

    else:

        model_results = (
            model_results.sort_values("MAE")
            .reset_index(drop=True)
        )

        best_model = model_results.iloc[0]

        if evaluation_type == "Extended":

            st.success(
                "Extended Evaluation: "
                "July 2024 – February 2025"
            )

        else:

            st.warning(
                "Showing original evaluation results. "
                "Extended results file was not found."
            )

        col1, col2, col3 = st.columns(3)

        col1.metric(
            "Best Model",
            best_model["Model"]
        )

        col2.metric(
            "Lowest MAE",
            f"{best_model['MAE']:,.2f}"
        )

        col3.metric(
            "Lowest RMSE",
            f"{model_results['RMSE'].min():,.2f}"
        )

        if evaluation_type == "Extended":

            col4, col5 = st.columns(2)

            col4.metric(
                "Months Evaluated",
                8
            )

            col5.metric(
                "Predictions Evaluated",
                208
            )

        st.divider()

        st.subheader(
            "📊 Mean Absolute Error"
        )

        fig_mae = px.bar(
            model_results,
            x="Model",
            y="MAE",
            color="Model",
            text="MAE",
            title="Forecasting Model MAE Comparison"
        )

        fig_mae.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )

        fig_mae.update_layout(
            showlegend=False,
            xaxis_title="Forecasting Model",
            yaxis_title="MAE (Units)"
        )

        st.plotly_chart(
            fig_mae,
            width="stretch"
        )

        st.subheader(
            "📈 Root Mean Squared Error"
        )

        fig_rmse = px.bar(
            model_results,
            x="Model",
            y="RMSE",
            color="Model",
            text="RMSE",
            title="Forecasting Model RMSE Comparison"
        )

        fig_rmse.update_traces(
            texttemplate="%{text:.2f}",
            textposition="outside"
        )

        fig_rmse.update_layout(
            showlegend=False,
            xaxis_title="Forecasting Model",
            yaxis_title="RMSE (Units)"
        )

        st.plotly_chart(
            fig_rmse,
            width="stretch"
        )

        st.subheader(
            "📋 Model Comparison Results"
        )

        st.dataframe(
            model_results,
            width="stretch",
            hide_index=True
        )

        if model_predictions is not None:

            st.subheader(
                "🔍 Historical Model Predictions"
            )

            st.dataframe(
                model_predictions,
                width="stretch",
                hide_index=True
            )

        st.success(
            f"{best_model['Model']} achieved "
            "the lowest MAE in the selected evaluation."
        )

        if evaluation_type == "Extended":

            st.caption(
                "The extended evaluation uses 208 "
                "category-month predictions across eight "
                "historical months. Results estimate "
                "one-month-ahead performance using "
                "observed historical lag values."
            )

        else:

            st.caption(
                "The original evaluation uses "
                "52 category-month observations "
                "across January–February 2025."
            )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "FashionIQ | AI Fashion Trends Intelligence "
    "& Forecasting System"
)