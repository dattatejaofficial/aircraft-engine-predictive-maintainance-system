import pandas as pd
import streamlit as st

from components.alert_panel import AlertPanel
from components.charts import DashboardCharts
from components.metrics_card import MetricCards

from services.api_client import api_client
from services.websocket_client import websocket_client

from state.dashboard_state import DashboardState

st.set_page_config(page_title="Engine Details", page_icon="🛠️", layout="wide")

st.title("🛠️ Engine Details")

DashboardState.initialize()
DashboardState.process_updates()

if not websocket_client.is_connected():
    websocket_client.start()

if not DashboardState.get_fleet():
    fleet = api_client.get_all_engines()
    DashboardState.set_fleet(fleet)

fleet = DashboardState.get_fleet()
engine_ids = sorted([int(engine_id) for engine_id in fleet['engines'].keys()])

default_engine = DashboardState.get_selected_engine()

if default_engine not in engine_ids:
    default_engine = engine_ids[0]

selected_engine = st.selectbox("Select Engine", options=engine_ids, index=engine_ids.index(default_engine))
DashboardState.set_selected_engine(selected_engine)

details = api_client.get_engine_details(selected_engine)

engine = details['current']
DashboardState.set_engine(engine)

history_df = pd.DataFrame(details['history'])

MetricCards.engine_summary(engine)

st.divider()

if engine.get('status') != 'ready':
    st.info(f"Engine {engine['engine_id']} is still collecting data ({engine.get('current_sequence_size', 0)}/30 )")
else:
    AlertPanel.show(
        alert_level=engine['alert_level'],
        alert_msg=engine['alert_message'],
        rul=engine['predicted_rul'],
        failure_prob=engine['failure_probability']
    )

st.divider()

sensor_columns = sorted([col for col in history_df.columns if col.startswith("sensor_")])
selected_sensor = st.selectbox("Select Sensor", sensor_columns)

DashboardCharts.sensor_trend(history_df, selected_sensor)

st.divider()

col1, col2 = st.columns(2)

with col1:
    DashboardCharts.rul_history(history_df)

with col2:
    DashboardCharts.failure_probability_history(history_df)