import asyncio

from src.agent import SimpleAgent
from src.llm import LLMClient


def test_agent_e2e_smoke():
    """
    Test end-to-end :
    - agent réel
    - llm mock
    - exécution complète
    """

    llm = LLMClient()  # LLM_MODE=mock via fixture
    agent = SimpleAgent(llm=llm)

    goal = "Rédiger un plan de déploiement d'une API backend"

    result = asyncio.run(
        agent.run(goal=goal, max_steps=3)
    )

    # Assertions E2E (comportement, pas implémentation)
    assert result.plan
    assert isinstance(result.plan, str)

    assert result.final_answer
    assert isinstance(result.final_answer, str)

    assert len(result.steps) > 0
    assert len(result.observations) > 0

def test_agent_uses_mock_llm():
    llm = LLMClient()
    agent = SimpleAgent(llm=llm)

    result = asyncio.run(
        agent.run("Test simple", max_steps=1)
    )

    assert "[MOCK]" in result.final_answer