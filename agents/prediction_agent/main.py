import asyncio
import time
import csv
from helper import row_to_formatted_string

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

# Settings can either be specified programmatically,
# or loaded from mcp_agent.config.yaml/mcp_agent.secrets.yaml
app = MCPApp(name="prediction-agent")  # settings=settings)

instruction = """
    You are an agent whose job is to predict whether or not a start-up will be an outlier success given the founder and startup profile:

    The data includes:
    - industry: The industry which the startup is operating in
    - ipos: Any previous IPOs by the founder
    - acquisitions: Any previous acquisitions by the founder
    - educations_json: Containing educational background of the founder in JSON
    - jobs_json: Containing professional background of the founder in JSON
    - anonymised_prose: Containing information on educational and professional background of the founder. 
    
    A start-up is considered successful if:
    - Exits via IPO at a valuation exceeding $500M;
    - Gets acquired for more than $500M;
    - Raises over $500M in total funding.

    Use sequential-thinking to facilitate a detailed, step-by-step thinking process for problem-solving and analysis, ensuring each decision is backed by the correct reasoning and tool calls are sequenced in the right order.
    
    The other servers include: 
    g-search to search for relevant information that could help with reasoning.
    fetcher to read the websites found by g-search and extract the necessary data.

    The output format should be:
    Yes/No 
    Explain the tool calls step by step, the reasoning behind them, and how this leds up to the final conclusion, all in one sentence.
"""

async def predictSuccess(prompt):
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context

        # logger.info("Current config:", data=context.config.model_dump())

        agent = Agent(
            name="prediction-agent",
            instruction=instruction,
            server_names=["g-search", "fetcher", "sequential-thinking"],
        )

        async with agent:
            # logger.info(prompt)
            # logger.info("prediction-agent: Connected to server, calling list_tools...")
            result = await agent.list_tools()
            # logger.info("Tools available:", data=result.model_dump())

            llm = await agent.attach_llm(OpenAIAugmentedLLM)
            result = await llm.generate(
                message=prompt,
            )
            for item in result:
                if item.content:
                    return item.content 


if __name__ == "__main__":
    start = time.time()
    ret = []
    with open("vcbench_final_public.csv", newline="", encoding="utf-8") as f_in:
        reader = csv.DictReader(f_in)
        for row in reader:
            ret.append((row.get("founder_uuid"), asyncio.run(predictSuccess(row_to_formatted_string(row)))))      
            if len(ret) >= 1:
                break      
    for (a,b) in ret:
        print(f"{a}: {b}")

    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")