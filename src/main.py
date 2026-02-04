from __future__ import annotations

import asyncio

from dotenv import load_dotenv

from src.agent import SimpleAgent
from src.llm import LLMClient

import logging
from src.logging_config import setup_logging

async def main() -> None:
    setup_logging(level=logging.DEBUG)

    logger = logging.getLogger("main")

    logger.info("Agent CLI started")

    load_dotenv()

    goal = input("Objectif: ").strip()
    if not goal:
        print("Objectif vide.")
        return

    agent = SimpleAgent(llm=LLMClient())
    result = await agent.run(goal)

    print("\n=== PLAN ===\n")
    print(result.plan)

    print("\n=== TRACE EXECUTION ===\n")
    for s in result.steps:
        print(s)
        print()

    print("\n=== REPONSE FINALE ===\n")
    print(result.final_answer)


if __name__ == "__main__":
    asyncio.run(main())
