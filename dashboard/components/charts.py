import pandas as pd
import plotly.express as px
import streamlit as st

class DashboardCharts:
    
    @staticmethod
    def fleet_alert_distribution(df : pd.DataFrame):
        if df.empty:
            st.info("No engine data available")
            return
        
        if 'alert_level' not in df.columns:
            st.info('No prediction data available yet. Waiting for engines to accumulate enough cycles...')
            return
        
        plot_df = df.dropna(subset=['alert_level'])
        if plot_df.empty:
            st.info("No prediction data available yet.")
            return
        
        counts = plot_df['alert_level'].value_counts().rename_axis('Alert').reset_index(name='Count')

        fig = px.pie(counts, names='Alert', values='Count', title='Fleet Alert Distribution')
        st.plotly_chart(fig, width='stretch')

    @staticmethod
    def fleet_rul_distribution(df: pd.DataFrame):
        if df.empty:
            st.info("No engine data available")
            return
        
        if 'predicted_rul' not in df.columns:
            st.info("No RUL predictions available yet. Waiting for engines to accumulate enough cycles...")
            return
        
        plot_df = df.dropna(subset=['predicted_rul'])
        if plot_df.empty:
            st.info("No RUL predictions available yet.")
            return
        
        fig = px.histogram(plot_df, x='predicted_rul', nbins=20, title='Predicted RUL Distribution')
        st.plotly_chart(fig, width='stretch')
    
    @staticmethod
    def sensor_trend(df: pd.DataFrame, sensor_name: str):
        if df.empty:
            st.info("No history available")
            return
        
        fig = px.line(df, x='cycle', y=sensor_name, title=f'{sensor_name} Trend')
        st.plotly_chart(fig, width='stretch')
    
    @staticmethod
    def rul_history(df: pd.DataFrame):
        if df.empty:
            st.info('No history available')
            return
        
        fig = px.line(df, x='cycle', y='predicted_rul', markers=True, title='Predicted RUL History')
        st.plotly_chart(fig, width='stretch')
    
    @staticmethod
    def failure_probability_history(df: pd.DataFrame):
        if df.empty:
            st.info('No history available')
            return

        fig = px.line(df, x='cycle', y='failure_probability', markers=True, title='Failure Probability History')
        st.plotly_chart(fig, width='stretch')