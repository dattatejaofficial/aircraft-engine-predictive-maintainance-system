import streamlit as st

from utils.constants import (
    SAFE,
    WARNING,
    CRITICAL,
    FAILURE_IMMINENT
)

class AlertPanel:
    _CONFIG = {
        SAFE : {
            "icon" : "🟢",
            "type" : "success"
        },
        WARNING : {
            "icon" : "🟡",
            "type" : "warning"
        },
        CRITICAL : {
            "icon" : "🟠",
            "type" : "critical"
        },
        FAILURE_IMMINENT : {
            "icon" : "🔴",
            "type" : "error"
        }
    }

    @staticmethod
    def show(alert_level: str, alert_msg : str, rul : int, failure_prob : float):
        config = AlertPanel._CONFIG.get(alert_level, AlertPanel._CONFIG[SAFE])

        body = (
            f"### {config['icon']} {alert_level.replace('_',' ')}\n\n"
            f"{alert_msg}\n\n"
            f"**Predicted RUL:** {rul} cycles\n\n"
            f"**Failure Probability:** {failure_prob * 100:.2f}%"
        )

        if config['type'] == "success":
            st.success(body)
        
        elif config['type'] == "warning":
            st.warning(body)
        
        else:
            st.error(body)