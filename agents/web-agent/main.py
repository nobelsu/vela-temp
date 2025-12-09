import asyncio
import time

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

# Settings can either be specified programmatically,
# or loaded from mcp_agent.config.yaml/mcp_agent.secrets.yaml
app = MCPApp(name="web-agent")  # settings=settings)

instruction = """
    You are an agent whose job is to identify the best Model Context Protocol server that falls under each of the categories provided. 
    
    Search and identify the best MCP server for each category listed, search for their documentation either online or on GitHub, and find their configuration.
    
    I want the result to be structured as a list of MCP servers in the .yaml format below.

    server_name:
      name: "server_name"
      description: "Description of server"
      command: "server_command"
      args: ["server_argument_1", "server_argument_2", ...]
      env: 
        environment_variable_1: "environment_variable_1"
        ...

    The server_name, command, args, and env can be obtained directly from the configuration. Meanwhile, generate a description of the server on your own.

    Only output the .yaml syntax above. Do not output anything else.
"""

async def findMCP(prompt):
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context

        logger.info("Current config:", data=context.config.model_dump())

        agent = Agent(
            name="web-agent",
            instruction=instruction,
            server_names=["g-search", "fetcher"],
        )

        async with agent:
            logger.info("web-agent: Connected to server, calling list_tools...")
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
    asyncio.run(findMCP("1. Web browsing/scraping \n 2. Payment processing \n 3. Website builder"))
    end = time.time()
    t = end - start

    print(f"Total run time: {t:.2f}s")