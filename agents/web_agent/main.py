import asyncio
import time

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

# Settings can either be specified programmatically,
# or loaded from mcp_agent.config.yaml/mcp_agent.secrets.yaml
app = MCPApp(name="web-agent")  # settings=settings)

instruction = """
    You are an agent whose job is to find MCP servers.
"""

async def findMCP(prompt):
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context

        logger.info("Current config:", data=context.config.model_dump())

        agent = Agent(
            name="web-agent",
            instruction=instruction,
            server_names=["mcp-compass", "sequential-thinking"],
        )

        async with agent:
            # logger.info("web-agent: Connected to server, calling list_tools...")
            result = await agent.list_tools()
            # logger.info("Tools available:", data=result.model_dump())

            llm = await agent.attach_llm(OpenAIAugmentedLLM)
            result = await llm.generate(
                message=prompt,
            )
            for item in result:
                if item.content:                    
                    logger.info(f"Results: {item.content}\n\n")


if __name__ == "__main__":
    start = time.time()
    asyncio.run(findMCP("1. Web browsing/scraping \n 2. Payment processing \n 3. Website builder"))
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")