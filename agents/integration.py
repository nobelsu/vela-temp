import asyncio
import time

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM

from agents.config.instructions import instructions_integration
from agents.config.servers import servers_integration

app = MCPApp(name="integration-agent")  

async def integrateMCP(prompt):
    async with app.run() as agent_app:
        logger = agent_app.logger
        context = agent_app.context

        agent = Agent(
            name="integration-agent",
            instruction=instructions_integration,
            server_names=servers_integration,
        )

        async with agent:
            logger.info("integration-agent: Connected to server!")

            llm = await agent.attach_llm(OpenAIAugmentedLLM)
            result = await llm.generate(
                message=prompt,
            )

            for item in result:
                if item.content:                    
                    logger.info(f"Results: {item.content}\n\n")