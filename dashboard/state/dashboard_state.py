from datetime import datetime
import streamlit as st

from utils.constants import SAFE, WARNING, CRITICAL, FAILURE_IMMINENT

from state.update_queue import update_queue

class DashboardState:

    @staticmethod
    def initialize():
        defaults = {
            "fleet_data" : {},
            "selected_engine" : None,
            "engine_data" : {},
            "alerts" : [],
            "backend_health" : {},
            "dashboard_metrics" : {},
            "last_updated" : None,
            "batch_prediction" : None
            # "ws_connected" : False
        }

        for key, val in defaults.items():
            if key not in st.session_state:
                st.session_state[key] = val
    
    @staticmethod
    def compute_dashboard_metrics():
        fleet = st.session_state.fleet_data
        engines = fleet.get('engines', {})
        
        total = len(engines)

        healthy = 0
        warning = 0
        critical = 0
        imminent = 0

        rul_vals = []

        for engine in engines.values():
            alert = engine.get("alert_level", "SAFE")
            rul = engine.get("predicted_rul")

            if rul is not None:
                rul_vals.append(rul)
            
            if alert == SAFE:
                healthy += 1
            elif alert == WARNING:
                warning += 1
            elif alert == CRITICAL:
                critical += 1
            elif alert == FAILURE_IMMINENT:
                imminent += 1
        
        avg_rul = sum(rul_vals) / len(rul_vals) if rul_vals else 0

        st.session_state.dashboard_metrics = {
            "total_engines" : total,
            "healthy_engines" : healthy,
            "warning_engines" : warning,
            "critical_engines" : critical + imminent,
            "average_rul" : round(avg_rul, 2)
        }
    
    @staticmethod
    def get_dashboard_metrics():
        return st.session_state.dashboard_metrics
    
    @staticmethod
    def set_fleet(data: dict):
        st.session_state.fleet_data = data
        st.session_state.last_updated = datetime.now()

        DashboardState.compute_dashboard_metrics()
    
    @staticmethod
    def get_fleet():
        return st.session_state.fleet_data
    
    @staticmethod
    def set_selected_engine(engine_id: int):
        st.session_state.selected_engine = engine_id
    
    @staticmethod
    def get_selected_engine():
        return st.session_state.selected_engine
    
    @staticmethod
    def set_engine(data: dict):
        st.session_state.engine_data = data
    
    @staticmethod
    def get_engine():
        return st.session_state.engine_data
    
    @staticmethod
    def get_alerts():
        fleet = st.session_state.fleet_data
        return [engine for engine in fleet.get('engines',{}).values() if engine.get('alert_level') not in (None, 'SAFE')]
    
    @staticmethod
    def get_grouped_alerts():
        grouped = {
            "FAILURE_IMMINENT" : [],
            "CRITICAL" : [],
            "WARNING" : []
        }

        for engine in DashboardState.get_alerts():
            grouped[engine['alert_level']].append(engine)
        
        for engines in grouped.values():
            engines.sort(key = lambda x: (x['predicted_rul'], -x['failure_probability']))
        
        return grouped
    
    @staticmethod
    def set_backend_health(data):
        st.session_state.backend_health = data
    
    @staticmethod
    def get_backend_health():
        return st.session_state.backend_health
    
    @staticmethod
    def set_dashboard_health(metrics: dict):
        st.session_state.dashboard_metrics = metrics
        
    @staticmethod
    def get_last_updated():
        return st.session_state.last_updated
    
    @staticmethod
    def set_batch_prediction(result):
        st.session_state.batch_prediction = result
    
    @staticmethod
    def get_batch_prediction():
        return st.session_state.batch_prediction
    
    # @staticmethod
    # def set_ws_status(status: bool):
    #     st.session_state.ws_connected = status
    
    # @staticmethod
    # def get_ws_status():
    #     return st.session_state.ws_connected
    
    @staticmethod
    def process_updates():
        if update_queue.empty():
            return
        
        fleet = DashboardState.get_fleet()

        if 'engines' not in fleet:
            fleet['engines'] = {}

        while not update_queue.empty():
            update = update_queue.get_nowait()

            if update.get('event') != 'engine_updates':
                continue

            data = update['data']
            fleet['engines'][str(update['engine_id'])] = data
        
        fleet['total_engines'] = len(fleet['engines'])
        
        DashboardState.set_fleet(fleet)
    
    @staticmethod
    def clear():
        keys = list(st.session_state.keys())

        for key in keys:
            del st.session_state[key]
        
        DashboardState.initialize()
        DashboardState.process_updates()