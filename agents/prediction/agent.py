from pydantic import BaseModel
import asyncio

from mcp_agent.app import MCPApp
from mcp_agent.agents.agent import Agent
from mcp_agent.workflows.llm.augmented_llm_openai import OpenAIAugmentedLLM
from mcp_agent.workflows.parallel.parallel_llm import ParallelLLM

# DO NOT CHANGE START

class PredictionResponse(BaseModel):
    prediction: bool
    reason: str
    thought_process: list[str]

async def predictSuccess(prompts):
    app = MCPApp(name="prediction-agent", settings="/agents/prediction/mcp_agent.config.yaml")

# DO NOT CHANGE END

    async with app.run() as agent_app:
        prediction_instruction = """
            You are an agent whose job is to predict whether a start-up will be an outlier success, based on founder profiles, team dynamics, and economic contexts.
            
            A start-up is considered successful if:
            - Exits via IPO at a valuation exceeding $500M;
            - Gets acquired for more than $500M;
            - Raises over $500M in total funding.

            Use sequential-thinking to facilitate a meticulous, step-by-step problem-solving process, ensuring each decision aligns with a comprehensive analysis of educational, economic, and team-related data. If needed, leverage general data exploration tools to enrich understanding of market contexts.
            
            The other servers include: 
            g-search to search for relevant information that could help with reasoning.
            fetcher to read the websites found by g-search and extract the necessary data.
        """

        agent = Agent(
            name="prediction-agent",
            instruction=prediction_instruction,
            server_names=["g-search", "fetcher", "sequential-thinking"],
        )

        async with agent:
            llm = await agent.attach_llm(OpenAIAugmentedLLM)
            results = []
            for prompt in prompts:
                result = await llm.generate_structured(message=prompt, response_model=PredictionResponse)
                results.append(result)
            return results