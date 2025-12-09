import asyncio
import time

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

# Settings can either be specified programmatically,
# or loaded from mcp_agent.config.yaml/mcp_agent.secrets.yaml
app = MCPApp(name="integration-agent")  # settings=settings)

instruction = """
    You are a code editing agent whose job is to add and remove MCP servers from main.py and mcp_agent.config.yaml.
"""

async def integrateMCP(prompt):
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context

        logger.info("Current config:", data=context.config.model_dump())

        agent = Agent(
            name="integration-agent",
            instruction=instruction,
            server_names=["g-search", "fetcher", "sequential-thinking", "filesystem"],
        )

        async with agent:
            logger.info("integration-agent: Connected to server, calling list_tools...")
            result = await agent.list_tools()
            logger.info("Tools available:", data=result.model_dump())

            llm = await agent.attach_llm(OpenAIAugmentedLLM)
            result = await llm.generate(
                message=prompt,
            )
            for item in result:
                if item.content:                    
                    logger.info(f"Results: {item.content}\n\n")


if __name__ == "__main__":
    start = time.time()
    asyncio.run(integrateMCP("""
    Edit main.py and mcp_agent.config.yaml using edit_file from filesystem to include the following MCP servers:
        1. brave search MCP
        2. github MCP
    
    Find out how to add these MCP servers by checking online documentation and code examples.
    """))
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")