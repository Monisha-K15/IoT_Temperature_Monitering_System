import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import joblib
import warnings
warnings.filterwarnings("ignore")
 
from sklearn.ensemble import IsolationForest
from sklearn.metrics import confusion_matrix, classification_report

st.set_page_config(
    page_title = "IoT Temperature Monitor",
    page_icon = "🌡️",
    layout = "wide",
    initial_sidebar_state = "expanded"
)

#Load data
@st.cache_data
def load_data():
    df = pd.read_csv("sensor_data.csv")   # <-- your CSV file
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df["hour"] = df["timestamp"].dt.hour
    return df

    
#Load Saved ML Models
@st.cache_resource 
def load_models():
    try:
        scaler = joblib.load("models/scaler.pkl")
        iso_model = joblib.load("models/isolation_forest.pkl")
        return scaler, iso_model
    except:
        return None, None 

#Load everything
df = load_data()
scaler, model = load_models()

#IQR Detection on loaded data
df["iqr_anomaly"] = False
for  location in df["location"].unique():
    subset = df[df["location"] == location]["temperature"]
    Q1, Q3 = subset.quantile(0.25), subset.quantile(0.75)
    IQR = Q3 - Q1
    lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
    mask = (df["location"] == location) & (((df["temperature"]  < lower) | df["temperature"] > upper))
    df.loc[mask, "iqr_anomaly"] = True

#Run isolation forest prediction
if model is not None :
    X = df[["temperature", "humidity"]].values
    X_scaled = scaler.transform(X)
    df["if_anomaly"] = model.predict(X_scaled) == -1
else:
    df["if_anomaly"] = False
# Combine IQR and Isolation Forest anomalies
df["is_anomaly"] = df["iqr_anomaly"] | df["if_anomaly"]

#Sidebar
with st.sidebar:
    st.title("🌡️ IoT Monitor")
    st.markdown("** IoT Temperature Monitoring System**")
    st.divider()

    #Filter by location
    st.subheader("Filter Data")
    locations = ["All"] + list(df["location"].unique())
    selected_loc = st.selectbox("Select Location:", locations)

    #Filter by sensor
    sensors = ["All"] + list(df["sensor_id"].unique())
    selected_sens = st.selectbox("Select Sensor:",sensors)

    st.divider()
    st.subheader("Dataset Info")
    st.metric("Total Readings", f"{len(df) :,}")
    st.metric("Total Anomalies", f"{df['is_anomaly'].sum() :,}")
    st.metric("Anomaly Rate", f"{df['is_anomaly'].mean()*100:.1f}%")


#Apply filters
filtered_df = df.copy()
if selected_loc != "All":
    filtered_df = filtered_df[filtered_df["location"] == selected_loc]
if selected_sens != "All":
    filtered_df = filtered_df[filtered_df["sensor_id"] == selected_sens]

#Main title
st.title("IoT Temperature Monitoring Dashboard")
st.markdown("Real-time sensor data analysis and anomaly detection")
st.divider()

#Create 4 tabs
tab1,tab2,tab3,tab4 = st.tabs([
    "Overview",
    "EDA Visualizations",
    "Anomaly Analysis",
    "Predict New Reading"
])

with tab1:
    st.subheader("📊 System Overview")
 
    # ── KPI Row — 4 metric cards side by side ─────────
    # st.columns(4) creates 4 equal columns
    col1, col2, col3, col4 = st.columns(4)
 
    with col1:
        st.metric("Total Readings",
                  f"{len(filtered_df):,}",
                  delta=None)
 
    with col2:
        anomaly_count = filtered_df["is_anomaly"].sum()
        st.metric("Actual Anomalies",
                  f"{anomaly_count:,}",
                  delta=f"{anomaly_count/len(filtered_df)*100:.1f}% rate")
 
    with col3:
        avg_temp = filtered_df["temperature"].mean()
        st.metric("Avg Temperature",
                  f"{avg_temp:.1f} C")
 
    with col4:
        sensors_active = filtered_df["sensor_id"].nunique()
        st.metric("Sensors Active",
                  f"{sensors_active}")
 
    st.divider()
 
    # ── Per Sensor Summary Table ──────────────────────
    st.subheader("Per Sensor Summary")
    sensor_summary = filtered_df.groupby("sensor_id").agg(
        Location      = ("location",    "first"),
        Total_Readings= ("temperature", "count"),
        Avg_Temp      = ("temperature", "mean"),
        Max_Temp      = ("temperature", "max"),
        Anomalies     = ("is_anomaly",  "sum"),
    ).round(2).reset_index()
    st.dataframe(sensor_summary, use_container_width=True)


# ════════════════════════════════════════════════════════
# TAB 2 — EDA Visualisations
# ════════════════════════════════════════════════════════
with tab2:
    st.subheader("📈 Exploratory Data Analysis")
 
    # ── Row 1: Temperature Distribution + Time Series ─
    col1, col2 = st.columns(2)
 
    with col1:
        # Histogram per location using plotly
        fig1 = px.histogram(
            filtered_df,
            x     = "temperature",
            color = "location",
            nbins = 30,
            title = "Temperature Distribution by Location",
            barmode = "overlay",
            opacity = 0.7
        )
        st.plotly_chart(fig1, use_container_width=True)
 
    with col2:
        # Time series line chart
        fig2 = px.line(
            filtered_df,
            x     = "timestamp",
            y     = "temperature",
            color = "sensor_id",
            title = "Temperature Over Time per Sensor"
        )
        st.plotly_chart(fig2, use_container_width=True)
 
    # ── Row 2: Correlation Heatmap + Hourly Pattern ───
    col3, col4 = st.columns(2)
 
    with col3:
        # Correlation heatmap
        corr = filtered_df[["temperature","humidity","is_anomaly"]].corr()
        fig3 = px.imshow(
            corr,
            text_auto = ".2f",
            color_continuous_scale = "RdBu_r",
            title = "Correlation Heatmap"
        )
        st.plotly_chart(fig3, use_container_width=True)
 
    with col4:
        # Average temperature by hour
        hourly = filtered_df.groupby(
            ["hour","location"])["temperature"].mean().reset_index()
        fig4 = px.line(
            hourly,
            x     = "hour",
            y     = "temperature",
            color = "location",
            markers = True,
            title = "Avg Temperature by Hour of Day"
        )
        st.plotly_chart(fig4, use_container_width=True)


# ════════════════════════════════════════════════════════
# TAB 3 — Anomaly Analysis
# ════════════════════════════════════════════════════════
with tab3:
    st.subheader("🚨 Anomaly Detection Results")
 
    # ── Method Comparison KPIs ────────────────────────
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Actual Anomalies",
                  filtered_df["is_anomaly"].sum())
    with col2:
        st.metric("IQR Detected",
                  filtered_df["iqr_anomaly"].sum())
    with col3:
        st.metric("Isolation Forest Detected",
                  filtered_df["if_anomaly"].sum())
 
    st.divider()
 
    # ── Actual vs IQR vs IF on time series ───────────
    st.subheader("Actual Anomalies vs Detected")
 
    fig5 = go.Figure()
 
    # All readings as blue line
    fig5.add_trace(go.Scatter(
        x    = filtered_df["timestamp"],
        y    = filtered_df["temperature"],
        mode = "lines",
        name = "All Readings",
        line = dict(color="steelblue", width=1),
        opacity = 0.6
    ))
 
    # Actual anomalies as red dots
    actual = filtered_df[filtered_df["is_anomaly"]]
    fig5.add_trace(go.Scatter(
        x      = actual["timestamp"],
        y      = actual["temperature"],
        mode   = "markers",
        name   = "Actual Anomaly",
        marker = dict(color="red", size=6)
    ))
 
    # IQR detected as purple dots
    iqr_det = filtered_df[filtered_df["iqr_anomaly"]]
    fig5.add_trace(go.Scatter(
        x      = iqr_det["timestamp"],
        y      = iqr_det["temperature"],
        mode   = "markers",
        name   = "IQR Detected",
        marker = dict(color="purple", size=6, symbol="triangle-up")
    ))
 
    # IF detected as orange dots
    if_det = filtered_df[filtered_df["if_anomaly"]]
    fig5.add_trace(go.Scatter(
        x      = if_det["timestamp"],
        y      = if_det["temperature"],
        mode   = "markers",
        name   = "IF Detected",
        marker = dict(color="orange", size=6, symbol="square")
    ))
 
    fig5.update_layout(title="All 3 Methods on Time Series")
    st.plotly_chart(fig5, use_container_width=True)
 
    # ── Per Sensor Anomaly Bar Chart ──────────────────
    st.subheader("Anomaly Count per Sensor")
    sensor_anom = filtered_df.groupby("sensor_id").agg(
        Actual = ("is_anomaly",   "sum"),
        IQR    = ("iqr_anomaly",  "sum"),
        IF     = ("if_anomaly",   "sum"),
    ).reset_index()
 
    fig6 = px.bar(
        sensor_anom.melt(id_vars="sensor_id",
                         var_name="Method",
                         value_name="Count"),
        x       = "sensor_id",
        y       = "Count",
        color   = "Method",
        barmode = "group",
        title   = "Anomaly Count per Sensor — All Methods"
    )
    st.plotly_chart(fig6, use_container_width=True)

# ═════════════════════
# TAB 4 — Predict New Reading
# ════════════════════════════════════════════════════════
with tab4:
    st.subheader("🔮 Predict: Normal or Anomaly?")
    st.markdown("Enter a sensor reading to predict if it is an anomaly.")
    st.divider()
 
    col1, col2 = st.columns(2)
 
    with col1:
        # Number input widgets
        temperature = st.number_input(
            "Temperature (C)",
            min_value = -10.0,
            max_value = 60.0,
            value     = 21.0,
            step      = 0.1
        )
        humidity = st.number_input(
            "Humidity (%)",
            min_value = 0.0,
            max_value = 100.0,
            value     = 50.0,
            step      = 0.1
        )
        location = st.selectbox(
            "Location",
            ["Server Room", "Warehouse", "Office"]
        )
 
    with col2:
        st.markdown("### Reference Ranges")
        st.info("""
        Server Room: 18-22C (anomaly >22C)
        Warehouse  : 15-25C (anomaly >25C)
        Office     : 20-26C (anomaly >26C)
        """)
 
    st.divider()
 
    # ── Predict Button ────────────────────────────────
    if st.button("🔮 Predict", type="primary",
                 use_container_width=True):
 
        # Method 1: IQR prediction
        loc_data = df[df["location"] == location]["temperature"]
        Q1 = loc_data.quantile(0.25)
        Q3 = loc_data.quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        iqr_result = temperature < lower or temperature > upper
 
        # Method 2: Isolation Forest prediction
        if model is not None:
            input_data   = np.array([[temperature, humidity]])
            input_scaled = scaler.transform(input_data)
            if_result    = model.predict(input_scaled)[0] == -1
        else:
            if_result = None
 
        # ── Show Results ──────────────────────────────
        col1, col2 = st.columns(2)
 
        with col1:
            st.markdown("### IQR Method")
            if iqr_result:
                st.error("🚨 ANOMALY DETECTED")
                st.write(f"Temperature {temperature}C is outside")
                st.write(f"normal range: {lower:.1f}C to {upper:.1f}C")
            else:
                st.success("✅ NORMAL READING")
                st.write(f"Temperature {temperature}C is within")
                st.write(f"normal range: {lower:.1f}C to {upper:.1f}C")
 
        with col2:
            st.markdown("### Isolation Forest")
            if if_result is None:
                st.warning("Model not loaded")
            elif if_result:
                st.error("🚨 ANOMALY DETECTED")
                st.write("Isolation Forest flagged this")
                st.write("reading as anomalous")
            else:
                st.success("✅ NORMAL READING")
                st.write("Isolation Forest classified")
                st.write("this as a normal reading")
