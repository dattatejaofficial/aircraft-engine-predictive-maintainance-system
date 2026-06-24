class EngineStateStore:
    def __init__(self):
        self._states = {}
    
    def update(self, engine_id: int, state: dict):
        self._states[engine_id] = state
    
    def get(self, engine_id: int):
        return self._states.get(engine_id)

    def get_all(self):
        return self._states

engine_state_store = EngineStateStore()