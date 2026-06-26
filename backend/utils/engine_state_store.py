from collections import deque
from collections import defaultdict

class EngineStateStore:
    def __init__(self, max_history: int = 500):
        self._latest_states : dict[int, dict] = {}
        self._history : dict[int, deque] = defaultdict(lambda: deque(maxlen=max_history))
    
    def update(self, engine_id: int, state: dict):
        self._latest_states[engine_id] = state
        self._history[engine_id].append(state.copy())
    
    def get(self, engine_id: int):
        return self._latest_states.get(engine_id)

    def get_all(self):
        return self._latest_states
    
    def get_history(self, engine_id: int):
        return list(self._history.get(engine_id, []))

    def get_all_history(self):
        return {
            engine_id : list(history) for engine_id, history in self._history.items()
        }
    
    def exists(self, engine_id: int) -> bool:
        return engine_id in self._latest_states
    
    def clear(self, engine_id: int):
        self._latest_states.pop(engine_id, None)
        self._history.clear()
    
    def clear_all(self):
        self._latest_states.clear()
        self._history.clear()

engine_state_store = EngineStateStore()