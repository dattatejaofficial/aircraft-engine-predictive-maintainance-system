from collections import deque
import pandas as pd

class SequenceBuffer:
    def __init__(self, sequence_length: int):
        self.sequence_length = sequence_length
        self.buffers : dict[int, deque] = {}
        self.baseline_map : dict[int, dict] = {}
        self.last_cycle : dict[int, int] = {}

    def add_record(self, engine_id: int, row: dict):
        current_cycle = row['cycle']

        if engine_id in self.last_cycle and current_cycle <= self.last_cycle[engine_id]:
            self.clear(engine_id)

        if engine_id not in self.buffers and row['cycle']:
            self.buffers[engine_id] = deque(maxlen=self.sequence_length)
        
        self.buffers[engine_id].append(row)
        self.last_cycle[engine_id] = current_cycle
    
    def set_baseline_map(self, engine_id: int, baseline_map: dict):
        self.baseline_map[engine_id] = baseline_map
    
    def get_baseline_map(self, engine_id: int) -> dict:
        return self.baseline_map.get(engine_id)

    def has_baseline_map(self, engine_id: int) -> bool:
        return engine_id in self.baseline_map
    
    def is_ready(self, engine_id: int) -> bool:
        return (engine_id in self.buffers and len(self.buffers[engine_id]) >= self.sequence_length)
    
    def current_size(self, engine_id: int) -> int:
        if engine_id not in self.buffers:
            return 0
        
        return len(self.buffers[engine_id])
    
    def remaining_records(self, engine_id: int) -> int:
        return max(0, self.sequence_length - self.current_size(engine_id))
    
    def get_sequence(self, engine_id: int) -> pd.DataFrame:
        return pd.DataFrame(list(self.buffers[engine_id]))
    
    def clear(self, engine_id: int) -> None:
        self.buffers.pop(engine_id, None)
        self.baseline_map.pop(engine_id, None)
        self.last_cycle.pop(engine_id, None)
    
    def clear_all(self) -> None:
        self.buffers.clear()
        self.baseline_map.clear()
        self.last_cycle.clear()