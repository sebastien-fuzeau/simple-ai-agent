from __future__ import annotations

from typing import Any, Dict, List

from src.llm import LLMClient
from src.prompts import SYSTEM_PROMPT, make_execution_prompt, make_planning_prompt
from src.types import AgentResult


class SimpleAgent:
    """
    Agent minimal: plan -> execute step-by-step -> final.
    Pas d'outils, pas de RAG, juste la boucle agentique.
    """

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def plan(self, goal: str) -> str:
        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_planning_prompt(goal)},
        ]
        return await self.llm.chat(messages)

    async def run(self, goal: str, max_steps: int = 6) -> AgentResult:
        plan_text = await self.plan(goal)

        steps: List[str] = []
        observations: List[str] = []

        # Exécution étape par étape
        for i in range(1, max_steps + 1):
            exec_prompt = make_execution_prompt(plan_text, i, observations)
            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": exec_prompt},
            ]
            out = await self.llm.chat(messages)

            steps.append(f"Step {i}: {out}")

            # Extraction observation “simple” (MVP)
            observations.append(out)

            # Heuristique : si le modèle dit “terminé”, on stoppe
            if "TERMIN" in out.upper() or "DONE" in out.upper() or "FIN" in out.upper():
                break

        # Réponse finale : on demande une synthèse
        final_messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": (
                        f"Objectif: {goal}\n\nPlan:\n{plan_text}\n\n"
                        f"Exécution (logs):\n" + "\n\n".join(steps) + "\n\n"
                                                                      "Donne la réponse finale claire en 5-10 lignes."
                ),
            },
        ]
        final_answer = await self.llm.chat(final_messages)

        return AgentResult(
            final_answer=final_answer,
            plan=plan_text,
            steps=steps,
            observations=observations,
        )
