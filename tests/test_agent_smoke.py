import asyncio

from src.agent import SimpleAgent
from src.llm import LLMClient


def test_agent_runs_in_mock_mode():
    llm = LLMClient(mode="mock")
    agent = SimpleAgent(llm=llm)

    result = asyncio.run(agent.run("Ecrire un plan de tests", max_steps=2))

    assert result.plan
    assert result.final_answer
