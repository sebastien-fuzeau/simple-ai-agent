from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional

Role = Literal["system", "user", "assistant"]

Message = Dict[str, Any]


@dataclass
class AgentResult:
    final_answer: str
    plan: str
    steps: List[str]
    observations: List[str]
