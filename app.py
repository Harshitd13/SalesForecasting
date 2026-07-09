import streamlit as st
import pandas as pd
import numpy as np
import json
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Sales Forecasting Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS
# ============================================================================
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700&family=Lato:wght@300;400;700&display=swap');

    .main {
        background: linear-gradient(135deg, #FFF8E7 0%, #FFE4B5 100%);
    }

    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #FF6B35 0%, #F7931E 100%);
    }

    [data-testid="stSidebar"] * {
        color: white !important;
    }

    [data-testid="stMetricValue"] {
        color: #2C3E50 !important;
        font-size: 28px !important;
        font-weight: 700 !important;
    }

    [data-testid="stMetricLabel"] {
        color: #34495E !important;
        font-weight: 600 !important;
        font-size: 16px !important;
    }

    [data-testid="stMetricDelta"] {
        color: #27AE60 !important;
        font-weight: 600 !important;
    }

    div[data-testid="metric-container"] {
        background: white;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        border-left: 4px solid #FF6B35;
    }

    h1 { color: #2C3E50 !important; font-weight: 700 !important; }
    h2 { color: #34495E !important; font-weight: 600 !important; }
    h3 { color: #2C3E50 !important; font-weight: 600 !important; }
    h4 { color: #34495E !important; font-weight: 600 !important; }
    p, span, div, label { color: #2C3E50 !important; }

    [data-testid="stRadio"] label {
        color: white !important;
        font-weight: 500 !important;
    }

    .stButton > button {
        background: linear-gradient(90deg, #FF6B35 0%, #F7931E 100%);
        color: white;
        font-weight: 600;
        border: none;
        border-radius: 8px;
        padding: 10px 24px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }

    .stButton > button:hover {
        background: linear-gradient(90deg, #F7931E 0%, #FF6B35 100%);
        box-shadow: 0 6px 8px rgba(0,0,0,0.15);
    }

    .stAlert {
        border-radius: 8px;
        border-left-width: 4px;
    }

    [data-testid="stDataFrame"] {
        background: white;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Fix: dropdown/selectbox text was black-on-black before this fix */
    [data-baseweb="select"] { background: white !important; }
    [data-baseweb="select"] > div {
        background-color: white !important;
        color: #2C3E50 !important;
    }
    [data-baseweb="select"] input { color: #2C3E50 !important; }
    [role="listbox"] { background-color: white !important; }
    [role="option"] {
        background-color: white !important;
        color: #2C3E50 !important;
    }
    [role="option"]:hover {
        background-color: #FFE4B5 !important;
        color: #2C3E50 !important;
    }

    .stMarkdown { color: #2C3E50 !important; }

    a { color: #FF6B35 !important; font-weight: 600 !important; }
    a:hover { color: #F7931E !important; }

    code {
        background-color: #FFF3E0 !important;
        color: #D84315 !important;
        padding: 2px 6px;
        border-radius: 4px;
    }

    [data-testid="stExpander"] {
        background: white;
        border-radius: 8px;
        border: 1px solid #FFE4B5;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# CHART SIZING CONSTANTS  (single source of truth -> no more mismatched heights)
# ============================================================================
BAR_HEIGHT = 400
LINE_HEIGHT = 450
FORECAST_HEIGHT = 550
SCATTER_HEIGHT = 500
BAR_MARGIN = dict(l=60, r=40, t=60, b=80)
LINE_MARGIN = dict(l=70, r=60, t=80, b=70)
Y_AXIS_PADDING = 1.3  # 30% headroom above tallest bar so text labels never get clipped

WARM_COLORS = ['#FF6B35', '#F7931E', '#FDC830', '#F37335', '#FF8C42']


def style_plotly_chart(fig):
    """Apply consistent styling to every Plotly chart in the app."""
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white',
        font=dict(color='#1A1A1A', size=13, family='Arial, sans-serif'),
        title_font=dict(size=18, color='#1A1A1A'),
        hoverlabel=dict(bgcolor="white", font_size=13, font_family="Arial", font_color='#1A1A1A'),
        legend=dict(
            font=dict(size=12, color='#1A1A1A'),
            bgcolor='rgba(255,255,255,0.9)',
            bordercolor='#2C3E50',
            borderwidth=1
        )
    )
    fig.update_xaxes(
        showgrid=True, gridwidth=1, gridcolor='#E8E8E8',
        showline=True, linewidth=2, linecolor='#1A1A1A',
        title_font=dict(color='#000000', size=15), tickfont=dict(color='#000000', size=12),
        mirror=True, ticks="outside", tickwidth=2, tickcolor='#1A1A1A', ticklen=6
    )
    fig.update_yaxes(
        showgrid=True, gridwidth=1, gridcolor='#E8E8E8',
        showline=True, linewidth=2, linecolor='#1A1A1A',
        title_font=dict(color='#000000', size=15), tickfont=dict(color='#000000', size=12),
        mirror=True, ticks="outside", tickwidth=2, tickcolor='#1A1A1A', ticklen=6
    )
    return fig


def make_bar_chart(x, y, y_title, text_fmt="${:,.0f}", color=None):
    """Standard bar chart: same height/margin/padding every time it's called."""
    max_y = max(y) if len(y) > 0 else 1
    fig = go.Figure(data=[go.Bar(
        x=x, y=y,
        marker_color=color if color else WARM_COLORS[:len(x)],
        text=[text_fmt.format(v) for v in y],
        textposition='outside',
        textfont=dict(color='#2C3E50', size=12)
    )])
    fig.update_layout(
        height=BAR_HEIGHT,
        margin=BAR_MARGIN,
        yaxis_title=y_title,
        yaxis=dict(range=[0, max_y * Y_AXIS_PADDING]),
        showlegend=False
    )
    return style_plotly_chart(fig)


def make_line_chart(x, y, y_title, name="Sales", height=LINE_HEIGHT):
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=x, y=y, mode='lines+markers', name=name,
        line=dict(color='#FF6B35', width=3),
        marker=dict(size=6, color='#F7931E'),
        fill='tozeroy', fillcolor='rgba(255, 107, 53, 0.2)'
    ))
    fig.update_layout(
        height=height, margin=LINE_MARGIN,
        xaxis_title='Date', yaxis_title=y_title, hovermode='x unified'
    )
    return style_plotly_chart(fig)


# ============================================================================
# LOAD DATA
# ============================================================================
@st.cache_data
def load_analysis_results():
    """Load pre-computed analysis results exported from the Jupyter notebook."""
    try:
        with open('dashboard_data.json', 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        return None
    except json.JSONDecodeError:
        return None


analysis_data = load_analysis_results()

if analysis_data is None:
    st.error("❌ **dashboard_data.json not found or unreadable.**")
    st.info("Run the export cell at the end of your notebook first, then place "
            "`dashboard_data.json` in the same folder as `app.py`.")
    st.stop()

dataset_info = analysis_data.get('dataset_info', {})
total_sales = dataset_info.get('total_sales', 0)

# ============================================================================
# SIDEBAR
# ============================================================================
st.sidebar.title("📊 Dashboard Navigation")
st.sidebar.markdown("---")

page = st.sidebar.radio(
    "Choose a page:",
    ["🏠 Home", "📈 Sales Overview", "🔮 Forecasts",
     "🚨 Anomaly Detection", "🎯 Product Segmentation", "📊 Category Analysis"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("### 📋 Dataset Summary")

st.sidebar.markdown(f"""
<div style='background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px; margin-bottom: 10px;'>
    <p style='margin: 0; font-size: 14px; color: white !important;'><b>📦 Transactions</b></p>
    <p style='margin: 0; font-size: 24px; font-weight: bold; color: white !important;'>{dataset_info.get('total_transactions', 0):,}</p>
</div>
""", unsafe_allow_html=True)

date_range = dataset_info.get('date_range', {})
st.sidebar.markdown(f"""
<div style='background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px; margin-bottom: 10px;'>
    <p style='margin: 0; font-size: 14px; color: white !important;'><b>📅 Date Range</b></p>
    <p style='margin: 0; font-size: 15px; font-weight: bold; color: white !important;'>{date_range.get('start', 'N/A')}</p>
    <p style='margin: 0; font-size: 15px; font-weight: bold; color: white !important;'>to {date_range.get('end', 'N/A')}</p>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown(f"""
<div style='background: rgba(255,255,255,0.2); padding: 15px; border-radius: 8px;'>
    <p style='margin: 0; font-size: 14px; color: white !important;'><b>💰 Total Sales</b></p>
    <p style='margin: 0; font-size: 24px; font-weight: bold; color: white !important;'>${total_sales/1e6:.2f}M</p>
</div>
""", unsafe_allow_html=True)

# ============================================================================
# HOME PAGE
# ============================================================================
if page == "🏠 Home":
    st.title("📊 Sales Forecasting & Demand Intelligence System")
    st.markdown("<h3 style='color: #FF6B35; margin-top: -10px;'>🎯 Advanced Analytics Dashboard</h3>",
                unsafe_allow_html=True)
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("📦 Total Orders", f"{dataset_info.get('total_orders', 0):,}", delta="4 Years of Data")

    with col2:
        monthly_sales_vals = analysis_data.get('monthly_sales', {}).get('sales', [])
        if len(monthly_sales_vals) > 12:
            first_year_avg = np.mean(monthly_sales_vals[:12])
            last_year_avg = np.mean(monthly_sales_vals[-12:])
            growth = ((last_year_avg - first_year_avg) / first_year_avg) * 100 if first_year_avg else 0
        else:
            growth = 0
        st.metric("💰 Total Revenue", f"${total_sales/1e6:.2f}M", delta=f"{growth:.1f}% growth")

    with col3:
        st.metric("📊 Categories", dataset_info.get('num_categories', 0), delta="All Profitable")

    with col4:
        st.metric("🌍 Regions", dataset_info.get('num_regions', 0), delta="Nationwide")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div style='background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 4px solid #FF6B35;'>
        <h3 style='color: #2C3E50; margin-top: 0;'>📋 Tasks Completed</h3>
        <p style='color: #2C3E50; line-height: 1.8;'>
        ✅ <b>Task 1:</b> Data Exploration & Business Questions<br>
        ✅ <b>Task 2:</b> Time Series Decomposition & Stationarity<br>
        ✅ <b>Task 3:</b> 3 Forecasting Models (SARIMA, Prophet, XGBoost)<br>
        ✅ <b>Task 4:</b> Category-wise Forecasting<br>
        ✅ <b>Task 5:</b> Anomaly Detection (Isolation Forest)<br>
        ✅ <b>Task 6:</b> Product Segmentation (K-Means)<br>
        ✅ <b>Task 7:</b> Interactive Dashboard (This!)
        </p>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div style='background: white; padding: 25px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); border-left: 4px solid #F7931E;'>
        <h3 style='color: #2C3E50; margin-top: 0;'>🛠️ Technologies Used</h3>
        <p style='color: #2C3E50; line-height: 1.8;'>
        <b>Programming:</b> Python, pandas, numpy<br>
        <b>Time Series:</b> SARIMA, Facebook Prophet<br>
        <b>Machine Learning:</b> XGBoost, Isolation Forest, K-Means<br>
        <b>Visualization:</b> Plotly<br>
        <b>Dashboard:</b> Streamlit<br>
        <b>Deployment:</b> GitHub, Streamlit Cloud
        </p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("<h2 style='color: #2C3E50;'>📊 Monthly Sales Trend</h2>", unsafe_allow_html=True)

    monthly_dates = pd.to_datetime(analysis_data['monthly_sales']['dates'])
    monthly_sales_values = analysis_data['monthly_sales']['sales']
    st.plotly_chart(make_line_chart(monthly_dates, monthly_sales_values, 'Sales ($)'),
                     use_container_width=True)

    st.markdown("---")
    st.info("👈 **Use the sidebar** to navigate through different analysis sections!")

# ============================================================================
# SALES OVERVIEW PAGE
# ============================================================================
elif page == "📈 Sales Overview":
    st.title("📈 Sales Overview & Trends")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Total Sales", f"${total_sales:,.0f}")
    with col2:
        avg_order = total_sales / max(dataset_info.get('total_orders', 1), 1)
        st.metric("📊 Avg Order Value", f"${avg_order:,.2f}")
    with col3:
        st.metric("📦 Total Orders", f"{dataset_info.get('total_orders', 0):,}")
    with col4:
        st.metric("⏱️ Avg Shipping", f"{dataset_info.get('avg_shipping_days', 4.0):.1f} days")

    st.markdown("---")

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 📦 Sales by Category")
        if 'task1' in analysis_data:
            categories = analysis_data['task1']['category_sales']['categories']
            cat_sales = analysis_data['task1']['category_sales']['sales']
            st.plotly_chart(
                make_bar_chart(categories, cat_sales, 'Total Sales ($)', text_fmt="${:,.1f}K"),
                use_container_width=True
            )
        else:
            st.info("Category sales data not available.")

    with col2:
        st.markdown("### 🌍 Sales by Region")
        if 'regions' in analysis_data:
            regions = analysis_data['regions']['names']
            region_sales = analysis_data['regions']['sales']
            fig = go.Figure(data=[go.Pie(
                labels=regions, values=region_sales, hole=0.4,
                marker=dict(colors=WARM_COLORS[:len(regions)]),
                textfont=dict(size=13, color='white'),
                textinfo='label+percent'
            )])
            fig.update_layout(height=BAR_HEIGHT, margin=BAR_MARGIN, paper_bgcolor='white', showlegend=True,legend=dict(font=dict(color="black",size=13)))
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Region sales data not available.")

    st.markdown("---")
    st.markdown("### 📅 Monthly Sales Trend")
    monthly_dates = pd.to_datetime(analysis_data['monthly_sales']['dates'])
    monthly_sales_values = analysis_data['monthly_sales']['sales']
    st.plotly_chart(make_line_chart(monthly_dates, monthly_sales_values, 'Sales ($)'),
                     use_container_width=True)

    st.markdown("---")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### 🗓️ Sales by Month (Seasonality)")
        if 'monthly_patterns' in analysis_data:
            month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                           'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
            monthly_pattern = analysis_data['monthly_patterns']['avg_sales']
            st.plotly_chart(
                make_bar_chart(month_names, monthly_pattern, 'Average Sales ($)',
                                text_fmt="${:,.1f}K", color=WARM_COLORS[0]),
                use_container_width=True
            )
        else:
            st.info("Monthly pattern data not available.")

    with col2:
        st.markdown("### 📊 Sales by Day of Week")
        if 'day_of_week' in analysis_data:
            dow_days = analysis_data['day_of_week']['days']
            dow_values = analysis_data['day_of_week']['sales']
            st.plotly_chart(
                make_bar_chart(dow_days, dow_values, 'Total Sales ($)',
                                text_fmt="${:,.0f}", color=WARM_COLORS[1]),
                use_container_width=True
            )
        else:
            st.info("Day-of-week data not available. Re-run the export cell in your notebook.")

# ============================================================================
# FORECASTS PAGE — WITH INTERACTIVE EXPLORER (Fix E)
# ============================================================================
elif page == "🔮 Forecasts":
    st.title("🔮 Sales Forecasting Models Compared")

    st.markdown("""
    <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;'>
    <h3 style='color: #2C3E50; margin-top: 0;'>📊 Three Forecasting Approaches</h3>
    <p style='color: #2C3E50; line-height: 1.8;'>
    🔸 <b>SARIMA:</b> Statistical time series model with seasonal components<br>
    🔸 <b>Facebook Prophet:</b> Automated forecasting tool<br>
    🔸 <b>XGBoost:</b> Machine learning with lag features
    </p>
    </div>
    """, unsafe_allow_html=True)

    # -------- MODEL COMPARISON --------
    if 'task3' in analysis_data:
        st.markdown("### 🏆 Model Performance Comparison")
        st.markdown("<br>", unsafe_allow_html=True)

        models      = analysis_data['task3']['model_comparison']['models']
        mae_values  = analysis_data['task3']['model_comparison']['mae']
        rmse_values = analysis_data['task3']['model_comparison']['rmse']
        mape_values = analysis_data['task3']['model_comparison']['mape']

        c1, c2, c3 = st.columns(3)
        with c1:
            st.markdown("#### MAE ($)")
            st.plotly_chart(make_bar_chart(models, mae_values, 'MAE ($)', "${:,.2f}"),
                            use_container_width=True)
        with c2:
            st.markdown("#### RMSE ($)")
            st.plotly_chart(make_bar_chart(models, rmse_values, 'RMSE ($)', "${:,.2f}"),
                            use_container_width=True)
        with c3:
            st.markdown("#### MAPE (%)")
            st.plotly_chart(make_bar_chart(models, mape_values, 'MAPE (%)', "{:.2f}%"),
                            use_container_width=True)

        best_idx        = int(np.argmin(mape_values))
        best_model_name = models[best_idx]
        best_mape       = mape_values[best_idx]
        best_mae        = mae_values[best_idx]

        st.markdown("---")

        # -------- INTERACTIVE FORECAST EXPLORER --------
        st.markdown("### 🎛️ Interactive Forecast Explorer")
        st.markdown("<br>", unsafe_allow_html=True)

        e1, e2, e3 = st.columns([1, 1, 1])
        with e1:
            segment_type = st.selectbox("Select Segment Type", ["Category", "Region"])
        with e2:
            if segment_type == "Category":
                seg_options = list(analysis_data.get('task4_categories', {}).keys()) or \
                              analysis_data.get('categories', {}).get('names', [])
            else:
                seg_options = list(analysis_data.get('task4_regions', {}).keys()) or \
                              analysis_data.get('regions', {}).get('names', [])
            selected_segment = st.selectbox(f"Select {segment_type}", seg_options)
        with e3:
            horizon = st.slider("Forecast Horizon (months)", 1, 3, 3)

        # Pull correct forecast bundle
        seg_bundle = None
        if segment_type == "Category" and 'task4_categories' in analysis_data:
            seg_bundle = analysis_data['task4_categories'].get(selected_segment)
        elif segment_type == "Region" and 'task4_regions' in analysis_data:
            seg_bundle = analysis_data['task4_regions'].get(selected_segment)

        # Historical trace source
        if segment_type == "Category" and selected_segment in analysis_data.get('categories', {}).get('sales_by_category', {}):
            hist = analysis_data['categories']['sales_by_category'][selected_segment]
            hist_dates, hist_sales = pd.to_datetime(hist['dates']), hist['sales']
        else:
            hist_dates = pd.to_datetime(analysis_data['monthly_sales']['dates'])
            hist_sales = analysis_data['monthly_sales']['sales']

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hist_dates, y=hist_sales, mode='lines+markers',
            name=f'{selected_segment} Historical',
            line=dict(color='#FF6B35', width=3), marker=dict(size=6)
        ))

        if seg_bundle:
            f_dates    = pd.to_datetime(seg_bundle['dates'])[:horizon]
            f_actual   = seg_bundle['actual'][:horizon]
            f_forecast = seg_bundle['forecast'][:horizon]

            fig.add_trace(go.Scatter(
                x=f_dates, y=f_actual, mode='markers', name='Actual (Test)',
                marker=dict(size=14, color='green', symbol='circle',
                            line=dict(width=2, color='darkgreen'))
            ))
            fig.add_trace(go.Scatter(
                x=f_dates, y=f_forecast, mode='lines+markers',
                name=f'{best_model_name} Forecast',
                line=dict(color='#F7931E', width=3, dash='dash'),
                marker=dict(size=14, symbol='diamond', color='#F7931E',
                            line=dict(width=2, color='#FF6B35'))
            ))

            # Metrics for the selected slice
            arr_a, arr_f = np.array(f_actual), np.array(f_forecast)
            seg_mae  = float(np.mean(np.abs(arr_a - arr_f)))
            seg_rmse = float(np.sqrt(np.mean((arr_a - arr_f) ** 2)))
        else:
            seg_mae = seg_rmse = None

        fig.update_layout(
            title=dict(text=f'{selected_segment} — {horizon}-Month {best_model_name} Forecast',
                       font=dict(size=20, color='#2C3E50'), x=0.5, xanchor='center'),
            xaxis_title='Date', yaxis_title='Sales ($)', hovermode='x unified',
            height=FORECAST_HEIGHT, margin=dict(l=80, r=80, t=100, b=80),
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="center", x=0.5)
        )
        st.plotly_chart(style_plotly_chart(fig), use_container_width=True)

        # -------- METRICS BELOW CHART --------
        m1, m2, m3 = st.columns(3)
        with m1:
            st.metric(f"MAE — {selected_segment}",
                      f"${seg_mae:,.2f}" if seg_mae is not None else "N/A")
        with m2:
            st.metric(f"RMSE — {selected_segment}",
                      f"${seg_rmse:,.2f}" if seg_rmse is not None else "N/A")
        with m3:
            st.metric(f"Best Overall MAPE", f"{best_mape:.2f}%")

        # -------- GROWTH WINNER --------
        if 'task4_growth' in analysis_data:
            st.markdown("---")
            st.markdown("### 📈 Strongest Upcoming Growth")
            growth_dict = analysis_data['task4_growth']
            winner_name = analysis_data.get('task4_winner', max(growth_dict, key=growth_dict.get))
            growth_df = pd.DataFrame({
                'Segment': list(growth_dict.keys()),
                'Growth (%)': [f"{v:+.2f}%" for v in growth_dict.values()]
            }).sort_values('Growth (%)', ascending=False)
            st.dataframe(growth_df, use_container_width=True, hide_index=True)
            st.success(f"🏆 **{winner_name}** shows the strongest upcoming growth.")

        st.markdown("---")
        st.success(f"💡 **Best Overall Model:** {best_model_name} — MAPE {best_mape:.2f}%, MAE ${best_mae:,.2f}")
    else:
        st.warning("⚠️ Task 3 results not available. Complete forecasting in your notebook.")

# ============================================================================
# ANOMALY DETECTION PAGE
# ============================================================================
elif page == "🚨 Anomaly Detection":
    st.title("🚨 Anomaly Detection")

    st.markdown("""
    <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;'>
    <h3 style='color: #2C3E50; margin-top: 0;'>🔍 Isolation Forest Algorithm</h3>
    <p style='color: #2C3E50; line-height: 1.8;'>
    Results from the anomaly detection analysis.<br><br>
    <b>Anomalies could indicate:</b><br>
    🔸 Promotional events or bulk orders (high sales)<br>
    🔸 System downtime or data issues (low sales)<br>
    🔸 Unusual market conditions
    </p>
    </div>
    """, unsafe_allow_html=True)

    if 'task5' in analysis_data:
        num_anomalies = analysis_data['task5'].get('num_anomalies', 0)
        mean_sales = analysis_data['task5'].get('mean_sales', 0)
        all_dates = pd.to_datetime(analysis_data['task5']['all_dates'])
        all_sales = analysis_data['task5']['all_sales']
        is_anomaly_list = analysis_data['task5'].get('is_anomaly', [])

        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("📅 Total Days", f"{len(all_dates):,}")
        with col2:
            st.metric("🚨 Anomalies Found", f"{num_anomalies}")
        with col3:
            rate = (num_anomalies / len(all_dates) * 100) if len(all_dates) else 0
            st.metric("📊 Anomaly Rate", f"{rate:.2f}%")

        st.markdown("---")
        st.markdown("### 📊 Daily Sales with Anomalies")

        normal_dates, normal_sales, anomaly_dates, anomaly_sales = [], [], [], []
        for date, sales, is_anom in zip(all_dates, all_sales, is_anomaly_list):
            if is_anom:
                anomaly_dates.append(date)
                anomaly_sales.append(sales)
            else:
                normal_dates.append(date)
                normal_sales.append(sales)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=normal_dates, y=normal_sales, mode='markers',
                                  name='Normal Days', marker=dict(color='#FFB84D', size=5, opacity=0.6)))
        fig.add_trace(go.Scatter(x=anomaly_dates, y=anomaly_sales, mode='markers',
                                  name='Anomalies',
                                  marker=dict(color='#FF3333', size=12, symbol='x',
                                              line=dict(width=2, color='#CC0000'))))
        fig.add_hline(y=mean_sales, line_dash="dash", line_color="#27AE60", line_width=3,
                      annotation_text=f"Mean: ${mean_sales:,.0f}", annotation_position="right")
        fig.update_layout(xaxis_title='Date', yaxis_title='Daily Sales ($)',
                          hovermode='x unified', height=SCATTER_HEIGHT, margin=LINE_MARGIN)
        st.plotly_chart(style_plotly_chart(fig), use_container_width=True)

        st.markdown("---")
        st.markdown("### 🔝 Top 10 Most Anomalous Days")

        if analysis_data['task5'].get('anomaly_dates'):
            anom_df = pd.DataFrame({
                'Date': analysis_data['task5']['anomaly_dates'],
                'Sales': analysis_data['task5']['anomaly_sales']
            }).nlargest(10, 'Sales')
            anom_df['Sales'] = anom_df['Sales'].apply(lambda x: f"${x:,.2f}")
            anom_df.columns = ['📅 Date', '💰 Sales']
            st.dataframe(anom_df, use_container_width=True, hide_index=True)
    else:
        st.warning("⚠️ Anomaly detection results not available. Complete Task 5 in your "
                   "notebook and re-run the export cell.")

# ============================================================================
# PRODUCT SEGMENTATION PAGE
# ============================================================================
elif page == "🎯 Product Segmentation":
    st.title("🎯 Product Segmentation — K-Means Results")

    st.markdown("""
    <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 20px;'>
    <h3 style='color: #2C3E50; margin-top: 0;'>🧩 Product Segmentation Strategy</h3>
    <p style='color: #2C3E50; line-height: 1.8;'>
    Products grouped by:<br>
    🔸 <b>Total Sales</b> • 🔸 <b>Sales Volatility</b> • 🔸 <b>Purchase Patterns</b>
    </p>
    </div>
    """, unsafe_allow_html=True)

    if 'task6' in analysis_data:
        # Guard against missing/renamed keys so this never throws a KeyError again.
        task6 = analysis_data['task6']
        products = task6.get('products', task6.get('sub_categories', []))
        total_sales_vals = task6.get('total_sales', [])
        volatility = task6.get('volatility', [])
        clusters_list = task6.get('clusters', [])

        if products and clusters_list:
            cluster_counts = Counter(clusters_list)

            st.markdown("### 📊 Product Clusters")
            cols = st.columns(len(cluster_counts))
            for col, (cluster_name, count) in zip(cols, cluster_counts.items()):
                with col:
                    st.markdown(f"""
                    <div style='background: white; padding: 20px; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); height: 160px; border-left: 4px solid #FF6B35; text-align: center;'>
                        <h4 style='color: #2C3E50; margin-top: 0;'>{cluster_name}</h4>
                        <p style='font-size: 32px; font-weight: bold; color: #FF6B35; margin: 10px 0;'>{count}</p>
                        <p style='color: #2C3E50; font-size: 14px; margin: 0;'>Products</p>
                    </div>
                    """, unsafe_allow_html=True)

            st.markdown("---")
            st.markdown("### 📈 Cluster Visualization")

            cluster_df = pd.DataFrame({
                'Product': products,
                'Total_Sales': total_sales_vals,
                'Sales_Volatility': volatility,
                'Cluster': clusters_list
            })

            unique_clusters = sorted(cluster_df['Cluster'].unique())
            color_map = {c: WARM_COLORS[i % len(WARM_COLORS)] for i, c in enumerate(unique_clusters)}

            fig = px.scatter(
                cluster_df, x='Total_Sales', y='Sales_Volatility', color='Cluster',
                size='Total_Sales', hover_data=['Product'], color_discrete_map=color_map,
                title='Product Segmentation Map — K-Means Results'
            )
            fig.update_layout(xaxis_title='Total Sales ($)', yaxis_title='Sales Volatility',
                              height=SCATTER_HEIGHT, margin=LINE_MARGIN)
            st.plotly_chart(style_plotly_chart(fig), use_container_width=True)

            st.markdown("---")
            st.success(f"💡 **Clustering Complete:** {len(products)} products segmented into "
                       f"{len(unique_clusters)} distinct demand patterns based on sales volume and volatility.")
        else:
            st.warning("⚠️ Task 6 data is present but missing expected fields "
                       "(`products`/`clusters`). Check the export cell in your notebook.")
    else:
        st.warning("⚠️ Clustering results not available. Complete Task 6 in your notebook "
                   "and re-run the export cell.")

# ============================================================================
# CATEGORY ANALYSIS PAGE
# ============================================================================
elif page == "📊 Category Analysis":
    st.title("📊 Category-Wise Analysis")

    if 'categories' in analysis_data:
        available_categories = analysis_data['categories']['names']

        st.markdown("### 🔍 Select a Category to Analyze:")
        selected_category = st.selectbox("Category", options=available_categories,
                                          label_visibility="collapsed")

        cat_data = analysis_data['categories']['sales_by_category'][selected_category]
        st.markdown("---")

        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("💰 Total Sales", f"${cat_data['total_sales']:,.0f}")
        with col2:
            st.metric("📦 Total Orders", f"{cat_data['total_orders']:,}")
        with col3:
            avg_order_val = cat_data['total_sales'] / max(cat_data['total_orders'], 1)
            st.metric("📊 Avg Order Value", f"${avg_order_val:,.2f}")
        with col4:
            cat_sales_vals = cat_data['sales']
            if len(cat_sales_vals) > 12:
                first_year = np.mean(cat_sales_vals[:12])
                last_year = np.mean(cat_sales_vals[-12:])
                growth = ((last_year - first_year) / first_year) * 100 if first_year > 0 else 0
            else:
                growth = 0
            st.metric("📈 Growth Rate", f"{growth:.1f}%")

        st.markdown("---")
        st.markdown(f"### 📈 {selected_category} — Monthly Sales Trend")

        cat_dates = pd.to_datetime(cat_data['dates'])
        cat_sales = cat_data['sales']
        st.plotly_chart(make_line_chart(cat_dates, cat_sales, 'Sales ($)', name=selected_category),
                         use_container_width=True)
    else:
        st.warning("⚠️ Category data not available.")

# ============================================================================
# FOOTER
# ============================================================================
st.markdown("---")
st.markdown("""
<div style='text-align: center; padding: 20px; background: white; border-radius: 10px; box-shadow: 0 4px 6px rgba(0,0,0,0.1);'>
    <p style='color: #2C3E50; font-size: 16px; font-weight: 600; margin: 5px 0;'>📊 Sales Forecasting Dashboard</p>
    <p style='color: #FF6B35; font-size: 14px; margin: 5px 0;'>Built with Streamlit</p>
    <p style='color: #7F8C8D; font-size: 12px; margin: 5px 0;'>Data Science & Machine Learning | 2026</p>
</div>
""", unsafe_allow_html=True)