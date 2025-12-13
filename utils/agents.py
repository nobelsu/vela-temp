import time

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_google import GoogleAugmentedLLM
from mcp_agent.workflows.llm.augmented_llm import RequestParams

app = MCPApp(name="util-agents")

async def promptAgent(prompt, name, instruction, server_names=[], iter=10, model="gemini-2.5-flash"):
    start = time.time()
    async with app.run() as agent_app:
        logger = agent_app.logger

        agent = Agent(
            name=name,
            instruction=instruction,
            server_names=server_names,
        )

        async with agent:
            llm = await agent.attach_llm(GoogleAugmentedLLM)
            result = await llm.generate_str(
                message=prompt,
                request_params=RequestParams(
                    max_iterations=iter,
                    model=model
                ),
            )
            end = time.time()
            logger.info(f"[{name}] Worked for {end-start}")
            return result