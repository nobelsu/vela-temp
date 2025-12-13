import asyncio

from agents.main.improver_agent import train 
from utils.csv import split 

if __name__ == "__main__":
    split()
    asyncio.run(train())    