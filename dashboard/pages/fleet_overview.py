import pandas as pd
import streamlit as st

from services.api_client import api_client
from services.websocket_client import websocket_client

from state.dashboard_state import DashboardState

from components.metrics_card import MetricCards
from components.engine_table import EngineTable
from components.charts import DashboardCharts

st.set_page_config(page_title="Fleet Overview", page_icon="✈️", layout="wide")

st.title("✈️ Fleet Overview")

DashboardState.initialize()
DashboardState.process_updates()


if not websocket_client.is_connected():
    if st.button("Reconnect & Refresh"):
        fleet = api_client.get_all_engines()
        DashboardState.set_fleet(fleet)
        websocket_client.start()
        st.rerun()

if not DashboardState.get_fleet():
    fleet = api_client.get_all_engines()

    DashboardState.set_fleet(fleet)

fleet = DashboardState.get_fleet()
metrics = DashboardState.get_dashboard_metrics()

col1, col2 = st.columns([8, 1])

with col2:
    if st.button("🔄 Refresh"):
        fleet = api_client.get_all_engines()
        DashboardState.set_fleet(fleet)
        st.rerun()

MetricCards.fleet_summary(metrics)

st.divider()

st.subheader("Fleet Status")

EngineTable.render(fleet["engines"])

st.divider()

fleet_df = pd.DataFrame(fleet['engines'].values())
col1, col2 = st.columns(2)

with col1:
    DashboardCharts.fleet_alert_distribution(fleet_df)

with col2:
    DashboardCharts.fleet_rul_distribution(fleet_df)

last_updated = DashboardState.get_last_updated()

if last_updated:
    st.caption(f"Last Updated: {last_updated.strftime('%d-%m-%Y %H:%M:%S')}")