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

    # Exclude incomplete March 2025
    return df[df["Month"] < "2025-03-01"].copy()


@st.cache_data
def load_model_comparison():
    extended_path = Path("models/extended_model_comparison.csv")
    original_path = Path("models/model_comparison.csv")

    if extended_path.exists():
        return pd.read_csv(extended_path), "Extended Evaluation"

    if original_path.exists():
        return pd.read_csv(original_path), "Original Evaluation"

    return None, None


df = load_data()
model_results, evaluation_type = load_model_comparison()

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
        .sort_values("Units_Sold", ascending=False)
        .head(n)
    )

    result["Segment"] = (
        result["Category"] + " - " + result["Sub Category"]
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

    st.plotly_chart(fig, width="stretch")


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
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Units Sold",
        yaxis_title=""
    )

    st.plotly_chart(fig, width="stretch")


# =====================================================
# PAGE 1: DASHBOARD
# =====================================================

if page == "Dashboard":

    st.title("👗 FashionIQ")
    st.subheader("Fashion Intelligence Dashboard")

    st.write(
        "Explore fashion demand, category performance, "
        "and historical seasonal trends."
    )

    total_units = int(df["Units_Sold"].sum())
    segments = category_segments(df)
    months = df["Month"].nunique()

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Units Sold", f"{total_units:,}")
    col2.metric("Fashion Segments", segments)
    col3.metric("Months Analyzed", months)

    st.divider()

    st.subheader("📈 Monthly Fashion Sales")
    display_monthly_chart(
        df,
        "Historical Monthly Fashion Demand"
    )

    st.subheader("🏆 Best-Selling Fashion Segments")
    display_top_segments(df)

    st.subheader("🛍️ Sales Distribution")

    category_sales = (
        df.groupby("Category", as_index=False)["Units_Sold"]
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

    st.plotly_chart(fig, width="stretch")


# =====================================================
# PAGE 2: FASHION TRENDS
# =====================================================

elif page == "Fashion Trends":

    st.title("📈 Fashion Trends Intelligence")

    st.write(
        "Analyze fashion category demand, top-selling "
        "segments, and seasonal purchasing patterns."
    )

    selected_category = st.selectbox(
        "Select Fashion Category",
        ["All Categories"]
        + sorted(df["Category"].unique().tolist())
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

    st.subheader("📊 Monthly Demand")

    display_monthly_chart(
        filtered_df,
        "Monthly Fashion Demand Trends"
    )

    st.subheader("🏆 Top-Selling Segments")
    display_top_segments(filtered_df)

    st.subheader("🥧 Category Distribution")

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

    st.plotly_chart(fig, width="stretch")

    st.subheader("🌸 Seasonal Demand Analysis")

    seasonal = monthly_totals(filtered_df)
    seasonal["Month_Number"] = seasonal["Month"].dt.month
    seasonal["Month_Name"] = seasonal["Month"].dt.strftime("%B")

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

    st.plotly_chart(fig, width="stretch")

    st.caption(
        "Seasonal averages use complete months from "
        "January 2023 to February 2025."
    )

    st.subheader("📋 Historical Fashion Data")

    st.dataframe(
        filtered_df.sort_values(
            "Month",
            ascending=False
        ),
        width="stretch"
    )


# =====================================================
# PAGE 3: DEMAND FORECASTING
# =====================================================

elif page == "Demand Forecasting":

    st.title("🔮 Fashion Demand Forecasting")

    st.write(
        "Estimate monthly fashion demand using "
        "historical seasonal sales patterns."
    )

    st.info(
        "Forecasts use sales from the same calendar "
        "month in the previous year."
    )

    category = st.selectbox(
        "Select Fashion Category",
        sorted(df["Category"].unique().tolist())
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

    selected_month = forecast_options[selected_label]

    historical_lookup = selected_data.set_index(
        "Month"
    )["Units_Sold"]

    forecast_records = []

    for month in forecast_months:
        source_month = month - pd.DateOffset(years=1)

        if source_month in historical_lookup.index:
            forecast_records.append({
                "Forecast Month": month,
                "Reference Month": source_month,
                "Predicted Units": int(
                    historical_lookup.loc[source_month]
                )
            })

    forecast_table = pd.DataFrame(forecast_records)

    if forecast_table.empty:
        st.error(
            "Historical sales are unavailable "
            "for the selected category."
        )

    else:
        selected_forecast = forecast_table[
            forecast_table["Forecast Month"] == selected_month
        ]

        if selected_forecast.empty:
            st.error(
                "Forecast unavailable for this month."
            )

        else:
            predicted_units = int(
                selected_forecast["Predicted Units"].iloc[0]
            )

            historical_month = (
                selected_month - pd.DateOffset(years=1)
            )

            latest_actual = int(
                selected_data["Units_Sold"].iloc[-1]
            )

            col1, col2, col3 = st.columns(3)

            col1.metric("Forecast Month", selected_label)

            col2.metric(
                "Predicted Demand",
                f"{predicted_units:,} units"
            )

            col3.metric(
                "Latest Recorded Monthly Sales",
                f"{latest_actual:,} units"
            )

            st.caption(
                f"Prediction based on "
                f"{historical_month.strftime('%B %Y')} sales."
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
                        color="#2196F3",
                        width=3
                    ),
                    marker=dict(size=6)
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=forecast_table["Forecast Month"],
                    y=forecast_table["Predicted Units"],
                    mode="lines+markers",
                    name="Forecasted Demand",
                    line=dict(
                        color="#AB47BC",
                        width=3,
                        dash="dash"
                    ),
                    marker=dict(size=8)
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=[selected_month],
                    y=[predicted_units],
                    mode="markers",
                    name="Selected Forecast Month",
                    marker=dict(
                        color="#FF9800",
                        size=15,
                        line=dict(
                            color="white",
                            width=2
                        )
                    )
                )
            )

            fig.update_layout(
                title=(
                    f"{category} - {subcategory}: "
                    "Historical Sales vs Forecast"
                ),
                xaxis_title="Month",
                yaxis_title="Units Sold",
                hovermode="x unified",
                legend=dict(
                    orientation="h",
                    yanchor="bottom",
                    y=1.02,
                    xanchor="right",
                    x=1
                )
            )

            st.plotly_chart(fig, width="stretch")

            st.subheader("📊 Forecast Summary")

            total_forecast = int(
                forecast_table["Predicted Units"].sum()
            )

            average_forecast = (
                forecast_table["Predicted Units"].mean()
            )

            peak_row = forecast_table.loc[
                forecast_table["Predicted Units"].idxmax()
            ]

            col1, col2, col3 = st.columns(3)

            col1.metric(
                "12-Month Forecast Total",
                f"{total_forecast:,}"
            )

            col2.metric(
                "Average Monthly Demand",
                f"{average_forecast:,.0f}"
            )

            col3.metric(
                "Peak Demand Month",
                peak_row["Forecast Month"].strftime("%B %Y")
            )

            st.subheader("📋 Monthly Demand Forecast")

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
                label="📥 Download Forecast CSV",
                data=csv,
                file_name="fashioniq_forecast.csv",
                mime="text/csv"
            )

            st.warning(
                "These forecasts cover March 2025 to "
                "February 2026 using historical seasonal "
                "patterns. They are not current market "
                "forecasts and have not been validated "
                "against actual sales for those months."
            )


# =====================================================
# PAGE 4: MODEL PERFORMANCE — UPDATED
# =====================================================

elif page == "Model Performance":

    st.title("🧠 Forecasting Model Performance")

    st.write(
        "Compare forecasting models using "
        "historical sales and chronological evaluation."
    )

    if model_results is None:
        st.error(
            "Model comparison file not found. "
            "Run evaluate_models_extended.py first."
        )

    else:
        model_results = (
            model_results.sort_values("MAE")
            .reset_index(drop=True)
        )

        best_model = model_results.iloc[0]

        if evaluation_type == "Extended Evaluation":
            st.success(
                "Extended evaluation loaded successfully: "
                "July 2024 – February 2025."
            )
        else:
            st.info(
                "Showing the original two-month evaluation. "
                "Run evaluate_models_extended.py for "
                "the extended results."
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

        if "Months Tested" in model_results.columns:
            col4, col5 = st.columns(2)

            col4.metric(
                "Months Evaluated",
                int(best_model["Months Tested"])
            )

            col5.metric(
                "Predictions Evaluated",
                int(best_model["Predictions"])
            )

        st.divider()

        # -----------------------------------------
        # MAE COMPARISON
        # -----------------------------------------

        st.subheader("📊 Mean Absolute Error")

        fig_mae = px.bar(
            model_results,
            x="Model",
            y="MAE",
            color="Model",
            text="MAE",
            title="Forecasting Models — MAE Comparison"
        )

        fig_mae.update_traces(
            texttemplate="%{text:,.2f}",
            textposition="outside"
        )

        fig_mae.update_layout(
            showlegend=False,
            xaxis_title="Forecasting Model",
            yaxis_title="MAE (Units)"
        )

        st.plotly_chart(fig_mae, width="stretch")

        # -----------------------------------------
        # RMSE COMPARISON
        # -----------------------------------------

        st.subheader("📈 Root Mean Squared Error")

        fig_rmse = px.bar(
            model_results,
            x="Model",
            y="RMSE",
            color="Model",
            text="RMSE",
            title="Forecasting Models — RMSE Comparison"
        )

        fig_rmse.update_traces(
            texttemplate="%{text:,.2f}",
            textposition="outside"
        )

        fig_rmse.update_layout(
            showlegend=False,
            xaxis_title="Forecasting Model",
            yaxis_title="RMSE (Units)"
        )

        st.plotly_chart(fig_rmse, width="stretch")

        # -----------------------------------------
        # EVALUATION RESULTS
        # -----------------------------------------

        st.subheader("📋 Model Evaluation Results")

        st.dataframe(
            model_results,
            width="stretch",
            hide_index=True
        )

        # -----------------------------------------
        # MONTHLY MODEL PERFORMANCE
        # -----------------------------------------

        extended_predictions_path = Path(
            "models/extended_model_predictions.csv"
        )

        if (
            evaluation_type == "Extended Evaluation"
            and extended_predictions_path.exists()
        ):
            predictions_df = pd.read_csv(
                extended_predictions_path
            )

            predictions_df["Month"] = pd.to_datetime(
                predictions_df["Month"]
            )

            predictions_df["Absolute_Error"] = (
                predictions_df["Actual"]
                - predictions_df["Predicted"]
            ).abs()

            monthly_errors = (
                predictions_df.groupby(
                    ["Month", "Model"],
                    as_index=False
                )["Absolute_Error"]
                .mean()
            )

            st.subheader(
                "📉 Monthly Forecasting Error"
            )

            fig_monthly = px.line(
                monthly_errors,
                x="Month",
                y="Absolute_Error",
                color="Model",
                markers=True,
                title="Monthly MAE by Forecasting Model"
            )

            fig_monthly.update_layout(
                xaxis_title="Evaluation Month",
                yaxis_title="Mean Absolute Error (Units)"
            )

            st.plotly_chart(
                fig_monthly,
                width="stretch"
            )

            st.subheader(
                "🔍 Actual vs Predicted Demand"
            )

            selected_model = st.selectbox(
                "Select Model",
                model_results["Model"].tolist()
            )

            selected_predictions = predictions_df[
                predictions_df["Model"] == selected_model
            ].copy()

            fig_scatter = px.scatter(
                selected_predictions,
                x="Actual",
                y="Predicted",
                color="Category",
                hover_data=[
                    "Sub Category",
                    "Month"
                ],
                title=(
                    f"Actual vs Predicted Demand — "
                    f"{selected_model}"
                )
            )

            min_value = min(
                selected_predictions["Actual"].min(),
                selected_predictions["Predicted"].min()
            )

            max_value = max(
                selected_predictions["Actual"].max(),
                selected_predictions["Predicted"].max()
            )

            fig_scatter.add_trace(
                go.Scatter(
                    x=[min_value, max_value],
                    y=[min_value, max_value],
                    mode="lines",
                    name="Perfect Prediction",
                    line=dict(
                        color="gray",
                        dash="dash"
                    )
                )
            )

            fig_scatter.update_layout(
                xaxis_title="Actual Units Sold",
                yaxis_title="Predicted Units Sold"
            )

            st.plotly_chart(
                fig_scatter,
                width="stretch"
            )

            st.subheader(
                "📋 Detailed Forecast Predictions"
            )

            st.dataframe(
                selected_predictions.sort_values(
                    "Month",
                    ascending=False
                ),
                width="stretch",
                hide_index=True
            )

        st.success(
            f"{best_model['Model']} achieved the lowest "
            "MAE in the selected evaluation."
        )

        st.caption(
            "The extended evaluation uses 208 "
            "category-month predictions across eight "
            "historical months. Results estimate "
            "one-month-ahead performance using "
            "observed historical lag values."
        )


# =====================================================
# FOOTER
# =====================================================

st.divider()

st.caption(
    "FashionIQ | AI Fashion Trends Intelligence "
    "& Forecasting System"
)