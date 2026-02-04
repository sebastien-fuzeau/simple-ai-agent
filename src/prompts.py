from __future__ import annotations


SYSTEM_PROMPT = """Tu es un agent IA.
Tu dois:
1) proposer un plan en étapes courtes
2) exécuter étape par étape
3) rester factuel et sûr
4) donner une réponse finale claire.

Contraintes:
- Si une info est manquante, indique-le et propose une hypothèse explicite.
- Ne fabrique pas de sources.
"""


def make_planning_prompt(goal: str) -> str:
    return (
        "Objectif:\n"
        f"{goal}\n\n"
        "Donne un plan en 4 à 8 étapes maximum, numérotées, très concrètes."
    )


def make_execution_prompt(plan: str, step_index: int, previous_obs: list[str]) -> str:
    obs_block = "\n".join(f"- {o}" for o in previous_obs) if previous_obs else "(aucune)"
    return (
        "Plan:\n"
        f"{plan}\n\n"
        f"Tu es à l'étape {step_index}.\n"
        "Exécute uniquement cette étape maintenant.\n\n"
        f"Observations déjà obtenues:\n{obs_block}\n\n"
        "Réponds avec:\n"
        "1) ACTION (ce que tu fais)\n"
        "2) OBSERVATION (résultat obtenu)\n"
        "3) NEXT (si on continue ou si c'est terminé)\n"
    )
