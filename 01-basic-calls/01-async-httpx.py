import httpx
import asyncio

async def run():
    async with httpx.AsyncClient() as client:
        response = await client.get('https://httpbin.org/get')
        print(response.json())
        print("Async HTTP request completed.")

def main():
    print("Running async HTTP request...")
    asyncio.run(run())

main()
