import streamlit as st

class MetricCards:

    @staticmethod
    def fleet_summary(metrics: dict):
        """
        Expected Metrics:
        {
            "total_engines" : 100,
            "healthy_engines" : 82,
            "warning_engines" : 12,
            "critical_engines" : 6,
            "average_rul" : 74.8 
        }
        """

        col1, col2, col3, col4, col5 = st.columns(5)

        with col1:
            st.metric(label="Total Engines", value=metrics.get("total_engines",0))
        
        with col2:
            st.metric(label="🟢 Healthy", value=metrics.get("healthy_engines",0))
        
        with col3:
            st.metric(label="🟡 Warning", value=metrics.get("warning_engines",0))
        
        with col4:
            st.metric(label="🔴 Critical", value=metrics.get("critical_engines",0))
        
        with col5:
            st.metric(label="Avg RUL", value=round(metrics.get("average_rul",0), 2))
        
    @staticmethod
    def engine_summary(engine: dict):
        """
        {
            "engine_id" : 15,
            "cycle" : 180,
            "predicted_rul" : 22,
            "failure_probability" : 0.87
            "alert_level" : "CRITICAL"
        }
        """

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric(label="Engine", value=engine.get("engine_id","-"))
        
        with col2:
            st.metric(label="Cycle", value=engine.get("cycle","-"))
        
        with col3:
            st.metric(label="Predicted RUL", value=engine.get("predicted_rul",0))
        
        with col4:
            st.metric(label="Failure %", value=f"{engine.get('failure_probability', 0) * 100:.1f}")
        