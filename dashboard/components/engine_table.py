import pandas as pd
import streamlit as st

class EngineTable:
    
    @staticmethod
    def render(engines: dict, *, show_only_alerts: bool = False):
        if not engines:
            st.info("No engine data availables")
            return

        rows = []

        for eid, eng in engines.items():
            rows.append({
                "Engine ID" : eng.get('engine_id', eid),
                "Cycle" : eng.get("cycle"),
                "Predicted RUL" : eng.get("predicted_rul", 0),
                "Failure Probability (%)" : round(eng.get('failure_probability', 0) * 100, 2),
                "Alert Level" : eng.get("alert_level"),
                "Alert Message" : eng.get("alert_message")
            })
        
        df = pd.DataFrame(rows)

        if show_only_alerts:
            df = df[df['Alert Level'] != 'SAFE']
        
        if df.empty:
            st.success("No active alerts")
            return
        
        priority = {
            "FAILURE_IMMINENT" : 0,
            "CRITICAL" : 1,
            "WARNING" : 2,
            "SAFE" : 3
        }

        df['Priority'] = df['Alert Level'].map(priority)
        df = df.sort_values(['Priority','Predicted RUL'], ascending=[True, True]).drop(columns="Priority").reset_index(drop=True)

        st.data_editor(df, disabled = True, width='stretch', hide_index=True)
