import streamlit as st
import pandas as pd

from services.api_client import api_client
from services.websocket_client import websocket_client

from state.dashboard_state import DashboardState

from components.metrics_card import MetricCards
from components.charts import DashboardCharts

st.set_page_config(page_title="Aircraft Engine Predictive Dashboard", page_icon="✈️", layout="wide")

st.title("✈️ Aircraft Engine Predictive Maintenance System")

col1, col2 = st.columns([8, 1])

with col2:
    if st.button("🔄 Refresh"):
        health = api_client.get_health()
        fleet = api_client.get_all_engines()

        DashboardState.set_backend_health(health)
        DashboardState.set_fleet(fleet)

        st.rerun()

DashboardState.initialize()
DashboardState.process_updates()

if not websocket_client.is_connected():
    websocket_client.start()

health = api_client.get_health()
DashboardState.set_backend_health(health)

if not DashboardState.get_fleet():
    fleet = api_client.get_all_engines()
    DashboardState.set_fleet(fleet)

fleet = DashboardState.get_fleet()

metrics = DashboardState.get_dashboard_metrics()
fleet_df = pd.DataFrame(fleet.get("engines", {}).values())

st.subheader("System Status")

col1, col2, col3 = st.columns(3)

with col1:
    if health['status'] == 'healthy':
        st.success("Backend: Healthy")
    else:
        st.error("Backend: Error")

with col2:
    if websocket_client.is_connected():
        st.success("Websocket: Connected")
    else:
        st.warning("Websocket: Disconnected")

with col3:
    st.info(f"Model Version: {health['model_version']}")

last_updated = DashboardState.get_last_updated()

if last_updated:
    st.caption(f"Last Updated: {last_updated.strftime('%d-%m-%Y %H:%M:%S')}")

st.divider()

MetricCards.fleet_summary(metrics)

st.divider()

DashboardCharts.fleet_alert_distribution(fleet_df)

st.divider()

st.subheader("Recent Critical Alerts")
alerts = DashboardState.get_grouped_alerts()
critical_alerts = alerts['FAILURE_IMMINENT'] + alerts['CRITICAL']

if critical_alerts:
    for engine in critical_alerts[:5]:
        st.error(f"Engine {engine['engine_id']} • Cycle {engine['cycle']} • RUL {engine['predicted_rul']} • {engine['alert_level'].replace('_',' ')}")
else:
    st.success("No critical engines detected.")