from __future__ import annotations

import logging
from typing import Any, Dict, List

from src.llm import LLMClient
from src.prompts import SYSTEM_PROMPT, make_execution_prompt, make_planning_prompt
from src.types import AgentResult

logger = logging.getLogger("agent")

class SimpleAgent:
    """
    Agent minimal: plan -> execute step-by-step -> final.
    Pas d'outils, pas de RAG, juste la boucle agentique.
    """

    def __init__(self, llm: LLMClient) -> None:
        self.llm = llm

    async def plan(self, goal: str) -> str:
        logger.info("Planning started", extra={"goal": goal})

        messages: List[Dict[str, Any]] = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": make_planning_prompt(goal)},
        ]
        plan = await self.llm.chat(messages)
        logger.info("Planning finished")
        logger.debug("Plan content:\n%s", plan)
        return plan

    async def run(self, goal: str, max_steps: int = 6) -> AgentResult:
        logger.info(
            "Agent run started",
            extra={"goal": goal, "max_steps": max_steps},
        )

        plan_text = await self.plan(goal)

        steps: List[str] = []
        observations: List[str] = []

        # Exécution étape par étape
        for i in range(1, max_steps + 1):
            exec_prompt = make_execution_prompt(plan_text, i, observations)

            logger.info("Executing step %s", i)
            logger.debug("Execution prompt:\n%s", exec_prompt)

            messages = [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": exec_prompt},
            ]
            out = await self.llm.chat(messages)

            logger.debug("LLM output for step %s:\n%s", i, out)

            steps.append(f"Step {i}: {out}")

            # Extraction observation “simple” (MVP)
            observations.append(out)

            # Heuristique : si le modèle dit “terminé”, on stoppe
            if "TERMIN" in out.upper() or "DONE" in out.upper() or "FIN" in out.upper():
                logger.info("Agent stopped early at step %s", i)
                break

        # Réponse finale : on demande une synthèse
        logger.info("Final synthesis started")

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

        logger.info("Agent run finished")

        return AgentResult(
            final_answer=final_answer,
            plan=plan_text,
            steps=steps,
            observations=observations,
        )
