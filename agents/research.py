import asyncio
import time

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

from config.instructions import instructions_research
from config.servers import servers_research

app = MCPApp(name="research-agent") 

async def researchMCP(prompt):
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context

        agent = Agent(
            name="research-agent",
            instruction=instructions_research,
            server_names=servers_research,
        )

        async with agent:
            logger.info("research-agent: Connected to server!")

            llm = await agent.attach_llm(OpenAIAugmentedLLM)
            result = await llm.generate(
                message=prompt,
            )
            for item in result:
                if item.content:                    
                    logger.info(f"Results: {item.content}\n\n")