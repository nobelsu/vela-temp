import asyncio
import time

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

# Settings can either be specified programmatically,
# or loaded from mcp_agent.config.yaml/mcp_agent.secrets.yaml
app = MCPApp(name="research-agent")  # settings=settings)

instruction = """
    You are a researcher agent. 

    You are trying to find ways to improve the set of Model Context Protocol (MCP) servers provided to another AI agent that predicts whether or not a start up will succeed based on anonymized founder profile. 

    Use the sequential-thinking to work through this problem step by step.
    
    Use g-search and fetcher at steps where they are required.
    
    You will be provided a list of MCP servers currently used by this AI agent, as well as an evaluation of its performance.

    Make an updated list of MCP servers the AI agent should use. 

    The output should be a list in the format of:
    
    1. MCP Server 1
    2. MCP Server 2
    3. MCP Server 3
    ...

    This list should be limited to 5. Explain why each MCP server has been selected.

    The list MUST consist of MCP servers only. A quick way to verify if something is an MCP server is if it directly mentions any of the keywords: "MCP server", "Model Context Protocol server"
"""

async def researchMCP(prompt):
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context

        logger.info("Current config:", data=context.config.model_dump())

        agent = Agent(
            name="research-agent",
            instruction=instruction,
            server_names=["g-search", "fetcher", "sequential-thinking"],
        )

        async with agent:
            logger.info("research-agent: Connected to server, calling list_tools...")
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
    asyncio.run(researchMCP("""
        Current tools: 
        - File editing
                            
        Performance:
        Poor. Tool does not seem to be used.
    """))
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")