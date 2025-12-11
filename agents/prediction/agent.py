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

# DO NOT CHANGE END

async def predictSuccess(prompts):
    app = MCPApp(name="prediction-agent", settings="/agents/prediction/mcp_agent.config.yaml")

    async with app.run() as agent_app:
        founder_instruction = """
        You are a rigorous Data Analyst evaluating a startup founder's profile.
        Input data includes:
        - `educations_json`: List of degrees. Key metric: `qs_ranking` (Lower is better, <50 is Elite, 1 is Top Tier like MIT/Stanford).
        - `jobs_json`: Career history. Look for:
            - Founder roles (CTO, CEO, Founder, Co-Founder).
            - Duration of roles.
            - `company_size`.
        - `ipos`: List of previous IPOs. HUGE POSITIVE SIGNAL.
        - `acquisitions`: List of previous acquisitions. HUGE POSITIVE SIGNAL.
        - `anonymised_prose`: A summary.

        Your Task:
        Analyze the data for HARD SIGNALS of competence using `sequential-thinking`.
        Use `brave-search` or `g-search` to verify the prestige of universities or companies if `qs_ranking` is missing or if the company is not well known but sounds impressive.
        Use `github` tools (if available) to search for the founder's username or repositories if they have a technical background (Computer Science degree, CTO role) to validate their technical depth.
        Use `arxiv` to search for the founder's name or research in the field if they have a PhD or research background.
        Use `explorium` tools (if available) to validate the "Track Record" if specific company names are mentioned in the prose or inferred.

        Steps:
        1. **Education Analysis**: Check QS ranking. Rank 1 is ELITE. Rank < 50 is Strong. PhD/Masters in relevant field is a plus.
           - *Nuance*: If education is NOT elite, look for "Spiky" achievements in `jobs_json` (e.g., early promotions, founding companies young).
        2. **Role Analysis**: 
           - Look for "Founder", "Co-Founder", "CTO", "CEO". 
           - **Crucial**: "myself only employees" or small teams (<10) in a Founder/CTO role indicates **Entrepreneurial Grit** and **Agency**. Do NOT penalize for this. It shows they built something from scratch.
           - Check durations. Long tenure (>3y) shows commitment. Short stints (<2y) are okay if they were acquisitions or if the founder started a new company immediately.
        3. **Technical Validation** (if applicable):
           - If the founder is technical (CTO/Engineer), use `github` to search for them. High follower count, popular repos, or contributions to major projects are HUGE PLUSES.
        4. **Track Record**: Cross-reference `jobs_json` with `ipos` and `acquisitions`. 
           - If `ipos` or `acquisitions` lists are not empty, this is a massive green flag.
           - If `jobs_json` mentions "Acquired by", that counts too.
           - Use `explorium`'s `fetch-businesses-events` if you have a company name to check for exit events.
        5. **Founder-Market Fit**: Does their past work experience (`industry` in `jobs_json`) align with the current startup's `industry`?

        Output a structured analysis:
        - Track Record Score: (High/Medium/Low) - High if IPO/Acquisition exists.
        - Founder-Market Fit: (Yes/No/Partial)
        - Education Score: (Elite/High/Medium/Low) - Elite if QS < 10.
        - Grit/Entrepreneurial Signal: (High/Medium/Low) - High if they have founded before (even if small). "Myself only" = High Grit if duration > 1 yr.
        - Technical Signal: (Strong/Medium/Weak/N/A) - Based on GitHub or role history.
        - Key Strengths: (List max 3)
        - Key Risks: (List max 3 - Be specific).
        """
        founder_analyst = Agent(
            name="founder_analyst",
            instruction=founder_instruction,
            server_names=["g-search", "brave-search", "fetcher", "sequential-thinking", "time", "memory", "github", "explorium", "arxiv"],
        )

        market_instruction = """
        You are a Market Researcher.
        Input data includes:
        - `industry`: The specific industry of the startup.
        - `anonymised_prose`: Brief description.

        Your Task:
        Analyze the *Sector* potential using your knowledge and specialized search tools.
        Use `explorium` tools (`fetch-businesses-statistics`, `fetch-businesses-events`) to get real-world data on the industry.
        Use `hn-server` to check the sentiment and 'hype' around this industry/technology.
        Use `exa-search` for semantic market research (e.g., "market size of [Industry]", "competitors to [Description]") if available.
        Use `brave-search` or `g-search` for specific trends. Use `g-search` to specifically find 'venture capital exit multiples [Industry] 2023' or 'recent acquisitions [Industry]'.
        Use `finance` (if available) to check the stock performance of major public companies in this sector as a proxy for market health (e.g., "Get stock price history for [Leading Company]").
        Use `time` to know the current year for context.

        1. **Market Trends**: Is this industry growing? (e.g. AI, CleanTech vs. declining industries). Look for CAGR > 10%.
        2. **Venture Scale**: Do companies in this space typically exit for >$500M? (Software/BioTech = Yes, Service/Consulting = No).
           - Use `explorium` to find if there are many companies with high revenue or recent large funding rounds in this sector.
           - *Correction*: "Software Development" is generally High potential, but depends on the niche. Look for "SaaS" or "Platform" keywords. Be OPTIMISTIC about 'Software Development' unless there are clear signs of saturation or decline.
        3. **Macro**: Is the current environment favorable for this sector?

        Query format: "Venture capital exits [Industry] [Current Year]", "Market size [Industry] 2025", "Stock performance [Leading Company in Sector]".

        Output a structured analysis:
        - Market Attractiveness: (High/Medium/Low)
        - Exit Potential: (High/Medium/Low) - Can it reach $500M+ valuation?
        - Key Market Drivers:
        - Key Market Risks:
        """
        market_analyst = Agent(
            name="market_analyst",
            instruction=market_instruction,
            server_names=["g-search", "brave-search", "exa-search", "fetcher", "sequential-thinking", "time", "memory", "finance", "explorium", "hn-server"],
        )

        decision_instruction = """
        You are a Senior VC Partner making a final Yes/No decision.
        You have reports from the Founder Analyst and Market Analyst.

        Criteria for SUCCESS (Yes):
        - **Strongly Preferred**: High potential for >$500M exit (Market Exit Potential High/Medium).
        - **Spikes**: Look for *exceptional* strengths (Spikes) that outweigh weaknesses.
            - **Elite Education** (QS Rank 1-10) + Technical Background = YES.
            - **Previous Exit** (IPO/Acquisition) = YES.
            - **Deep Domain Expertise** (Years in same industry) + Founder Role = YES.
            - **High Grit + Technical Signal**: If a founder has "High" Grit (founded before) AND "Strong" Technical Signal (GitHub/Engineering roles), they can succeed even without Elite Education.
            - **Market Optimism**: TRUST the Market Analyst's 'Exit Potential'. If Market is High, lean YES.
            - **No Exit Penalty**: Do not penalize for 'No Exit' if the founder is young/first-time but has Grit.
        
        - **Don't Over-Penalize**: 
            - **Entrepreneurial Grit**: Small founder roles ("myself only") count as POSITIVE grit.
            - **Technical Founders**: Don't reject for "limited management experience" if they are technical.
            - **Market**: If Market is "Medium" but Founder is "Elite" OR has "High Grit", Lean YES.
            - **Track Record**: If no previous exit, but Elite Education or High Grit is present, do NOT automatically reject.
        
        - **Reasoning**:
            - If Founder has Elite Education OR Track Record OR High Grit/Fit -> Lean YES.
            - If Market is Low Attractiveness -> Lean NO (unless Founder is a Superstar).
            - **Correction for recent errors**: Do not reject a founder just because they lack an exit if they have Elite Education and High Grit.
            - **Correction**: "Software Development" is a High potential market.
        """
        decision_maker = Agent(
            name="decision_maker",
            instruction=decision_instruction,
            server_names=["sequential-thinking", "memory"],
        )

        parallel = ParallelLLM(
            fan_in_agent=decision_maker,
            fan_out_agents=[founder_analyst, market_analyst],
            llm_factory=OpenAIAugmentedLLM,
        )

        results = []
        for prompt in prompts:
            result = await parallel.generate_structured(message=prompt, response_model=PredictionResponse)
            results.append(result)
        return results

if __name__ == "__main__":
    asyncio.run(predictSuccess(["\n        industry: \"Software Development\",\n        ipos: [],\n        acquisition: [],\n        educations_json: [\n            {\n                        \"degree\": \"MSc\",\n                        \"field\": \"Microengineering\",\n                        \"qs_ranking\": \"200+\"\n            },\n            {\n                        \"degree\": \"BSc\",\n                        \"field\": \"Microtechnique\",\n                        \"qs_ranking\": \"200+\"\n            }\n],\n        jobs_json: [\n            {\n                        \"role\": \"Software Engineer\",\n                        \"company_size\": \"11-50 employees\",\n                        \"industry\": \"Technology, Information & Internet Platforms\",\n                        \"duration\": \"<2\"\n            },\n            {\n                        \"role\": \"Software Engineer\",\n                        \"company_size\": \"\",\n                        \"industry\": \"\",\n                        \"duration\": \"<2\"\n            }\n],\n        anonymised_prose: \"\"\"\n            This founder leads a startup in the Software Development industry.\nEducation:\n* MSc in Microengineering (Institution QS rank 200+)\n* BSc in Microtechnique (Institution QS rank 200+)\n\nProfessional experience:\n* Software Engineer for <2 years in the `Technology, Information & Internet Platforms` industry (11-50 employees)\n* Software Engineer for <2 years\n        \"\"\"\n        "]))