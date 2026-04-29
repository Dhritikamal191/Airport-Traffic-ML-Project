# ===============================
# IMPORTS
# ===============================
import streamlit as st
import pandas as pd
import pickle
import shap
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

st.markdown("""
<style>

/* ===============================
   🌙 MAIN APP BACKGROUND FIX
=============================== */
[data-testid="stAppViewContainer"] {
    background-color: #0e1117;
    color: white;
}

/* ===============================
   📦 MAIN CONTENT AREA
=============================== */
[data-testid="stHeader"] {
    background: transparent;
}

[data-testid="stToolbar"] {
    background: transparent;
}

/* ===============================
   📄 MAIN BLOCK (CONTENT)
=============================== */
.main {
    background-color: #0e1117;
}

/* ===============================
   📌 SIDEBAR
=============================== */
section[data-testid="stSidebar"] {
    background-color: #111827;
}

/* ===============================
   🔘 BUTTONS
=============================== */
.stButton>button {
    background: linear-gradient(90deg, #6366f1, #8b5cf6);
    color: white;
    border-radius: 10px;
    border: none;
    padding: 0.5em 1em;
}

/* ===============================
   📊 METRIC CARDS
=============================== */
[data-testid="stMetric"] {
    background-color: #1f2937;
    border-radius: 12px;
    padding: 15px;
    border: 1px solid #374151;
}

/* ===============================
   🔽 DROPDOWN FIX
=============================== */
div[data-baseweb="select"] {
    background-color: #1f2937 !important;
    color: white !important;
}

/* ===============================
   📋 EXPANDER
=============================== */
details {
    background-color: #1f2937;
    border-radius: 10px;
    padding: 10px;
}

/* ===============================
   📊 RADIO (TABS)
=============================== */
div[role="radiogroup"] label {
    background-color: #1f2937;
    border-radius: 8px;
    padding: 8px 16px;
}

/* ===============================
   📈 PLOTLY BACKGROUND
=============================== */
.js-plotly-plot {
    background-color: #0e1117 !important;
}

</style>
""", unsafe_allow_html=True)

# ===============================
# CONFIG
# ===============================
st.set_page_config(page_title="Airport Traffic Dashboard", layout="wide")

st.title("Airport Traffic Forecasting Dashboard")

# ===============================
# LOAD MODEL
# ===============================
@st.cache_resource
def load_model():
    with open("xgb_airport_pipeline.pkl", "rb") as f:
        return pickle.load(f)

model = load_model()

# ===============================
# LOAD DATA
# ===============================
@st.cache_data
def load_data():
    df = pd.read_csv("airport_traffic_2025.csv")
    df['FLT_DATE'] = pd.to_datetime(df['FLT_DATE'])

    # Feature engineering
    df['YEAR'] = df['FLT_DATE'].dt.year
    df['MONTH'] = df['FLT_DATE'].dt.month
    df['DAY'] = df['FLT_DATE'].dt.day
    df['WEEKDAY'] = df['FLT_DATE'].dt.weekday
    df['IS_WEEKEND'] = (df['WEEKDAY'] >= 5).astype(int)

    df['DEP_ARR_RATIO'] = df['FLT_DEP_1'] / (df['FLT_ARR_1'] + 1)
    df['IFR_RATIO'] = df['FLT_TOT_IFR_2'] / (df['FLT_TOT_1'] + 1)

    return df

df = load_data()

# ===============================
# SIDEBAR INPUT
# ===============================
st.sidebar.header("Input Parameters")
# ===============================
# Date Controls
# ===============================
with st.sidebar.expander("Date Controls", expanded=True):
    year = st.selectbox("Year", sorted(df['YEAR'].unique()))
    month = st.selectbox("Month", sorted(df['MONTH'].unique()))
    day = st.slider("Day", 1, 31, 15)

# ===============================
# Airport Selection
# ===============================
with st.sidebar.expander("Airport Info"):
    airport = st.selectbox("Airport (ICAO)", df['APT_ICAO'].unique())
    state = st.selectbox("State", df['STATE_NAME'].unique())

# ===============================
# Feature Controls
# ===============================
with st.sidebar.expander("Model Inputs"):
    weekday = st.selectbox("Weekday (0=Mon)", list(range(7)))
    is_weekend = 1 if weekday >= 5 else 0

    dep_arr_ratio = st.slider("Dep/Arr Ratio", 0.1, 3.0, 1.0)
    ifr_ratio = st.slider("IFR Ratio", 0.0, 1.0, 0.5)

input_df = pd.DataFrame({
    'YEAR':[year],'MONTH':[month],'DAY':[day],'WEEKDAY':[weekday],
    'IS_WEEKEND':[is_weekend],'APT_ICAO':[airport],'STATE_NAME':[state],
    'DEP_ARR_RATIO':[dep_arr_ratio],'IFR_RATIO':[ifr_ratio]
})

# ===============================
# PREDICTION + KPI
# ===============================
st.subheader("Prediction")

if st.button("Predict Traffic"):
    pred = model.predict(input_df)[0]

    col1, col2, col3 = st.columns(3)

    col1.metric("Predicted Flights", int(pred))
    col2.metric("Selected Month", month)
    col3.metric("Airport", airport)
# ===============================
# INSIGHTS
# ===============================
st.subheader("Traffic Insights")

col1, col2 = st.columns(2)

with col1:
     st.write("### Monthly Average Traffic")

     monthly = df.groupby('MONTH')['FLT_TOT_1'].mean().reset_index()
     fig = px.line(monthly,x='MONTH',y='FLT_TOT_1', markers=True,title="Monthly Average Flight Traffic")
     fig.update_layout(xaxis_title="Month",yaxis_title="Average Flights",template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
     fig.update_xaxes(showgrid=False)
     fig.update_yaxes(showgrid=False)
     st.plotly_chart(fig, use_container_width=True)

with col2:
     st.write("### Top 10 Airports")

     top_airports = (df.groupby('APT_ICAO')['FLT_TOT_1'].sum().nlargest(10).reset_index())
     fig = px.bar(top_airports,x='APT_ICAO',y='FLT_TOT_1',title="Top 10 Busiest Airports",text_auto=True)
     fig.update_layout(xaxis_title="Airport",yaxis_title="Total Flights",template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
     fig.update_xaxes(showgrid=False)
     fig.update_yaxes(showgrid=False)
     st.plotly_chart(fig, use_container_width=True)

# ===============================
# SHAP EXPLANATION
# ===============================
st.subheader("Model Explanation")

if "show_shap" not in st.session_state:
       st.session_state.show_shap = False


if st.button("Show SHAP Analysis"):
       st.session_state.show_shap = True

if st.session_state.show_shap:
    try:
        pre = model.named_steps['preprocessor']
        xgb = model.named_steps['model']

        # ===============================
        # TRANSFORM DATA
        # ===============================
        X_trans = pre.transform(input_df)

        # Background data for global SHAP
        X_sample = pre.transform(df.sample(200))

        # Feature names
        feature_names = pre.get_feature_names_out()

        # ===============================
        # SHAP EXPLAINER
        # ===============================
        explainer = shap.TreeExplainer(xgb)

        shap_values = explainer.shap_values(X_trans)
        shap_values_global = explainer.shap_values(X_sample)

        # ===============================
        # GLOBAL FEATURE IMPORTANCE
        # ===============================
        st.subheader("Global SHAP Feature Importance")

        shap_df = pd.DataFrame(shap_values_global, columns=feature_names)

        global_importance = pd.DataFrame({'Feature': feature_names,'Importance': abs(shap_df).mean().values}).sort_values(by='Importance', ascending=False)

        fig_global = px.bar(global_importance.head(15),x='Importance',y='Feature',orientation='h',color='Importance',color_continuous_scale='Blues',title="Top Features (Global Impact)")

        fig_global.update_layout(yaxis={'categoryorder':'total ascending'},template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        st.plotly_chart(fig_global, use_container_width=True)

        # ===============================
        # LOCAL EXPLANATION (WATERFALL STYLE)
        # ===============================
        st.subheader("Local Explanation (Waterfall Style)")

        local_vals = shap_values[0]

        waterfall_df = pd.DataFrame({'Feature': feature_names,'SHAP Value': local_vals}).sort_values(by='SHAP Value', key=abs, ascending=False).head(10)

        fig_waterfall = go.Figure(go.Bar(x=waterfall_df['SHAP Value'],y=waterfall_df['Feature'],orientation='h',marker=dict(color=waterfall_df['SHAP Value'],colorscale='RdBu')))

        fig_waterfall.update_layout(title="Feature Contribution (Positive vs Negative)",yaxis={'categoryorder':'total ascending'},template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig_waterfall, use_container_width=True)

        # ===============================
        # DEPENDENCE PLOT
        # ===============================
        # ===============================

        interaction_feature = st.selectbox("Select Interaction Feature (color)",feature_names,index=1)
        interaction_index = list(feature_names).index(interaction_feature)
        feature_index = list(feature_names).index(interaction_feature)
        dependence_df = pd.DataFrame({'Feature Value': X_sample[:, feature_index],'SHAP Value': shap_values_global[:, feature_index],'Interaction Feature': X_sample[:, interaction_index]})

        fig_dep = px.scatter(dependence_df,x='Feature Value',y='SHAP Value',color='Interaction Feature', color_continuous_scale='Viridis',title=f"Dependence Plot: {interaction_feature}",opacity=0.7,trendline="lowess")
       
        # ===============================
        # LAYOUT IMPROVEMENTS
        # ===============================
        fig_dep.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",xaxis_title=f"{interaction_feature} Value",yaxis_title="SHAP Impact",coloraxis_colorbar=dict(title=interaction_feature),title_x=0.3)

        # Better hover
        fig_dep.update_traces(marker=dict(size=6), hovertemplate="<b>Feature Value:</b> %{x}<br>" +"<b>SHAP Value:</b> %{y}<br>" +"<b>Interaction:</b> %{marker.color}<extra></extra>")
        fig.update_xaxes(showgrid=False)
        fig.update_yaxes(showgrid=False)
        st.plotly_chart(fig_dep, use_container_width=True)

    except Exception as e:
        st.error(f"SHAP failed: {e}")

# ===============================
# DATA VIEW
# ===============================
with st.sidebar.expander("View Dataset"):
    st.dataframe(df.head())

# ===============================
# NEXT 6 MONTHS FORECAST
# ===============================
import datetime

st.subheader("Next 6 Months Forecast")

if st.button("Generate Forecast"):

    # Start from selected year/month
    start_date = datetime.date(int(year), int(month), 1)

    future_data = []

    for i in range(6):
        future_month = (start_date.month + i - 1) % 12 + 1
        future_year = start_date.year + ((start_date.month + i - 1) // 12)

        # Assume mid-month day
        day = 15

        date_obj = datetime.date(future_year, future_month, day)

        weekday = date_obj.weekday()
        is_weekend = 1 if weekday >= 5 else 0

        future_data.append({
            'YEAR': future_year,
            'MONTH': future_month,
            'DAY': day,
            'WEEKDAY': weekday,
            'IS_WEEKEND': is_weekend,
            'APT_ICAO': airport,
            'STATE_NAME': state,
            'DEP_ARR_RATIO': dep_arr_ratio,
            'IFR_RATIO': ifr_ratio
        })

    future_df = pd.DataFrame(future_data)

    # Predict
    predictions = model.predict(future_df)

    future_df['Predicted Flights'] = predictions

    # ===============================
    # PLOT
    # ===============================
    fig = px.line(future_df,x='MONTH',y='Predicted Flights',markers=True,title="Next 6 Months Flight Forecast")
    fig.update_layout(xaxis_title="Month",yaxis_title= "Predicted Flights",template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
    fig.update_xaxes(showgrid=False)
    fig.update_yaxes(showgrid=False)
    st.plotly_chart(fig, use_container_width=True)

    # ===============================
    # TABLE
    # ===============================
    st.dataframe(future_df[['YEAR','MONTH','Predicted Flights']])

st.subheader("IFR vs Total Flights Comparison")

ifr_df = df.groupby(['APT_ICAO','MONTH'])[['FLT_TOT_1','FLT_TOT_IFR_2']].mean().reset_index()

fig = px.line(ifr_df,x='MONTH',y=['FLT_TOT_1','FLT_TOT_IFR_2'],markers=True,title="IFR vs Total Flights (Monthly)")
fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
st.plotly_chart(fig, use_container_width=True)

col1,col2=st.columns(2)

with col1:
     st.subheader("State-wise Traffic Distribution")

     state_df = df.groupby('STATE_NAME')['FLT_TOT_1'].sum().nlargest(10).reset_index()

     fig = px.bar(state_df,x='STATE_NAME',y='FLT_TOT_1',title= "Top States by Traffic",text_auto=True)
     fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
     fig.update_xaxes(showgrid=False)
     fig.update_yaxes(showgrid=False)
     st.plotly_chart(fig, use_container_width=True)

with col2:
     st.subheader("Weekend vs Weekly Traffic")

     week_df = df.groupby('IS_WEEKEND')['FLT_TOT_1'].mean().reset_index()

     week_df['Type'] = week_df['IS_WEEKEND'].map({0:'Weekday',1:'Weekend'})

     fig = px.bar(week_df,x='Type',y='FLT_TOT_1',title="Weekend vs Weekday Traffic")
     fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
     fig.update_xaxes(showgrid=False)
     fig.update_yaxes(showgrid=False)
     st.plotly_chart(fig, use_container_width=True)

st.subheader("Heatmap (Day vs Month)")

heat_df = df.pivot_table(values='FLT_TOT_1',index='MONTH', columns='DAY',aggfunc='mean')

fig = px.imshow(heat_df,aspect="auto",title="Traffic Heatmap (Year vs Month)")
fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
st.plotly_chart(fig, use_container_width=True)

X = df[['YEAR', 'MONTH', 'DAY', 'WEEKDAY', 'IS_WEEKEND','APT_ICAO', 'STATE_NAME','DEP_ARR_RATIO', 'IFR_RATIO']]

y = df['FLT_TOT_1']

df['Predicted'] = model.predict(X)

actual_vs_pred_df = df[['FLT_DATE', 'FLT_TOT_1', 'Predicted']].copy()

actual_vs_pred_df.rename(columns={'FLT_TOT_1': 'Actual'}, inplace=True)

fig = px.scatter(actual_vs_pred_df,x='Actual',y='Predicted',trendline="ols",title="Actual vs Predicted (Model Performance)",opacity=0.6)
# Add perfect prediction line
min_val = min(actual_vs_pred_df['Actual'].min(), actual_vs_pred_df['Predicted'].min())
max_val = max(actual_vs_pred_df['Actual'].max(), actual_vs_pred_df['Predicted'].max())
fig.add_shape(type="line",x0=min_val, y0=min_val,x1=max_val, y1=max_val,line=dict(dash="dash"))
fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",xaxis_title="Actual Flights",yaxis_title="Predicted Flights",title_x=0.3)
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
fig.update_traces(marker=dict(size=6),hovertemplate="<b>Actual:</b> %{x}<br>" +"<b>Predicted:</b> %{y}<extra></extra>")
st.plotly_chart(fig, use_container_width=True)

fig = px.line(actual_vs_pred_df,x='FLT_DATE',y=['Actual', 'Predicted'],title="Prediction Monitoring Over Time")
fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)")
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
st.plotly_chart(fig, use_container_width=True)

actual_vs_pred_df['Error'] = abs(actual_vs_pred_df['Actual'] - actual_vs_pred_df['Predicted'])

error_trend = actual_vs_pred_df.groupby(actual_vs_pred_df['FLT_DATE'].dt.month)['Error'].mean().reset_index()

# Rolling mean
actual_vs_pred_df['Rolling_Error'] = actual_vs_pred_df['Error'].rolling(5).mean()

fig = px.line(actual_vs_pred_df,x='FLT_DATE',y=['Error', 'Rolling_Error'],title="Error Trend Monitoring")

# Add alert threshold
threshold = actual_vs_pred_df['Error'].mean() * 1.5

fig.add_hline(y=threshold,line_dash="dash",annotation_text="Alert Threshold")

fig.update_layout(template="plotly_dark",paper_bgcolor="rgba(0,0,0,0)",plot_bgcolor="rgba(0,0,0,0)",xaxis_title="Date",yaxis_title="Error")
fig.update_xaxes(showgrid=False)
fig.update_yaxes(showgrid=False)
st.plotly_chart(fig, use_container_width=True)
