import asyncio
import random
import aiohttp
import pandas as pd

class EngineSimulator:
    def __init__(self, engine_id: int, engine_df: pd.DataFrame, api_url: str, semaphore):
        self.engine_id = engine_id
        self.engine_df = engine_df.sort_values('cycle')
        self.api_url = api_url
        self.semaphore = semaphore

    async def run(self, session: aiohttp.ClientSession):
        print(f'Engine {self.engine_id} started ({len(self.engine_df)} cycles)')

        for _, row in self.engine_df.iterrows():
            payload = row.to_dict()
            cycle = int(row['cycle'])
            try:
                await asyncio.sleep(random.uniform(0.5,2.0))
                async with self.semaphore:
                    async with session.post(self.api_url, json=payload) as response:
                        if response.status == 200:
                            result = await response.json()
                            print(f"ENGINE {self.engine_id} Cycle={cycle} Status={result.get('status')}")
                        else:
                            error = await response.text()
                            print(f"ENGINE {self.engine_id} Cycle={cycle} HTTP {response.status} {error}")

            except Exception as e:
                print(f'[E{self.engine_id}] Cycle={int(row["cycle"])} Error={e}')
                    
        print(f'Engine {self.engine_id} completed')