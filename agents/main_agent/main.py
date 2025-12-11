import asyncio
import time

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

# Settings can either be specified programmatically,
# or loaded from mcp_agent.config.yaml/mcp_agent.secrets.yaml
app = MCPApp(name="main-agent")  # settings=settings)

instruction = """
    You are an AI agent specialist. 

    You are trying to build the best prediction AI agent possible. You want to do this by adding or removing Model Context Protocol servers to and from the agent, as well as fine-tuning instructions.
    
    For context:
    
    This prediction AI agent predicts whether or not a start up will succeed based on:
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

    You have been provided with the following MCP servers:
    1. sequential-thinking: To break down this problem into subproblems and work on it step-by-step, sequentially
    2. g-search: To conduct research, to find out what MCP servers would be good to implement 
    3. fetcher: To fetch and extract data from websites that g-search finds
    4. mcp-compass and mcp-registry: To find more details on specific MCP servers if necessary
    5. filesystem: With access to the prediction_agent's directory, where you can directly read from the main.py and mcp_agent.config.yaml files to understand the current MCP servers being used and instructions, as well as edit them. 
    6. think-mcp: To think and reason before making decisions
    7. context7: To find documentation and code examples to integrate the MCP servers
    
    Your input will be the results from a previous run of the prediction agent. It will contain the accuracy of the agent, as well as the prediction agent's explanations for its mistakes.

    Ask for human input when confused or in need of extra information.
"""

async def iterateMCP(prompt):
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context

        logger.info("Current config:", data=context.config.model_dump())

        agent = Agent(
            name="main-agent",
            instruction=instruction,
            server_names=["sequential-thinking", "g-search", "fetcher",  "filesystem", "think-mcp", "mcp-compass", "mcp-registry", "context7"],
        )

        async with agent:
            logger.info("main-agent: Connected to server, calling list_tools...")
            result = await agent.list_tools()
            logger.info("Tools available:", data=result.model_dump())

            llm = await agent.attach_llm(OpenAIAugmentedLLM)
            result = await llm.generate(
                message=prompt,
            )
            for item in result:
                if item.content:                    
                    logger.info(f"Results: {item.content}\n\n")