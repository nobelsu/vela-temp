import agents.main.config as Config
from utils.agents import promptAgent

async def summarize(prompt):
    response = await promptAgent(prompt, "summarizer-agent", Config.instruct_summarize, Config.server_summarize)
    return response

async def extract(prompt):
    response = await promptAgent(prompt, "extractor-agent", Config.instruct_extract)
    return response