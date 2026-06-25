import asyncio
import aiohttp
import pandas as pd

from engine_simulator import EngineSimulator

API_URL = 'http://localhost:8000/predict/stream'
DATA_PATH = f'test_data/data.csv'

NUM_ENGINES = 5
MAX_CONCURRENT_REQUESTS = 20

async def main():
    df = pd.read_csv(DATA_PATH)
    engine_ids = sorted(df['engine_id'].unique())[:NUM_ENGINES]

    print(f'Creating {len(engine_ids)} simulators...')

    semaphore = asyncio.Semaphore(MAX_CONCURRENT_REQUESTS)
    timeout = aiohttp.ClientTimeout(total=60)

    async with aiohttp.ClientSession() as session:
        tasks = []

        for engine_id in engine_ids:
            engine_df = df[df['engine_id'] == engine_id]
            simulator = EngineSimulator(
                engine_id=engine_id,
                engine_df=engine_df,
                api_url=API_URL,
                semaphore=semaphore
            )

            tasks.append(asyncio.create_task(simulator.run(session)))

        await asyncio.gather(*tasks)
    
    print('Fleet Simulation Completed')

if __name__ == '__main__':
    asyncio.run(main())