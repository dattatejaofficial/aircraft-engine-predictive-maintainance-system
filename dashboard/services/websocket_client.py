import asyncio
import json
import threading

import websockets

from state.update_queue import update_queue
from utils.constants import WEBSOCKET_URL

class DashboardWebSocketClient:
    def __init__(self):
        self._thread = None
        self._running = False
        self._connected = False
    
    def _run(self):
        asyncio.run(self._listen())

    def _process_message(self, msg: str):
        payload = json.loads(msg)
        if payload.get('event') != 'engine_update':
            return
        
        update_queue.put(payload['data'])
        # fleet = DashboardState.get_fleet()

        # if 'engines' not in fleet:
        #     fleet['engines'] = {}
        
        # fleet['engines'][data['engine_id']] = data

        # DashboardState.set_fleet(fleet)

    async def _listen(self):
        while self._running:
            try:
                async with websockets.connect(WEBSOCKET_URL, ping_interval=20, ping_timeout=20) as websocket:
                    print("Connecting to Websocket")
                    # DashboardState.set_ws_status(True)
                    self._connected = True

                    async for msg in websocket:
                        self._process_message(msg)
            
            except Exception as e:
                self._connected = False
                # DashboardState.set_ws_status(False)
                print(f"WebSocket Error: {e}")
                await asyncio.sleep(5)

    def start(self):
        if self._running:
            return
        
        self._running = True
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
    
    def is_connected(self):
        return self._connected
    
websocket_client = DashboardWebSocketClient()