import streamlit as st

from components.alert_panel import AlertPanel

from services.api_client import api_client
from services.websocket_client import websocket_client

from state.dashboard_state import DashboardState

st.set_page_config(page_title="Critical Alerts", page_icon="🚨", layout="wide")

st.title("🚨 Critical Alerts")

DashboardState.initialize()
DashboardState.process_updates()

if not websocket_client.is_connected():
    websocket_client.start()

if not DashboardState.get_fleet():
    fleet = api_client.get_all_engines()
    DashboardState.set_fleet(fleet)

col1, col2 = st.columns([8, 1])

with col2:
    if st.button('🔄 Refresh'):
        fleet = api_client.get_all_engines()
        DashboardState.set_fleet(fleet)

        st.rerun()

alerts = DashboardState.get_grouped_alerts()

sections = [
    ("FAILURE_IMMINENT", "🔴 Failure Imminent"),
    ("CRITICAL", "🟠 Critical"),
    ("WARNING", "🟡 Warning")
]

for key, title in sections:
    engines = alerts[key]

    if not engines:
        continue

    st.subheader(f"{title} ({len(engines)})")

    for engine in engines:
        AlertPanel.show(
            alert_level=engine['alert_level'],
            alert_msg=engine['alert_message'],
            rul=engine['predicted_rul'],
            failure_prob=engine['failure_probability']
        )
    
        with st.expander("Engine Details"):
            col1, col2 = st.columns(2)

            with col1:
                st.write(f"**Engine ID:** {engine['engine_id']}")
                st.write(f"**Cycle:** {engine['cycle']}")

            with col2:
                st.write(f"**Predicted RUL:** {engine['predicted_rul']}")
                st.write(f"**Failure Probability:** {engine['failure_probability']*100:.2f}")

    st.divider()

if not any(alerts.values()):
    st.success("No Active Alerts in the Fleet")