from src.prompts import make_execution_prompt, make_planning_prompt


def test_planning_prompt_contains_goal():
    goal = "Créer une checklist de release"
    p = make_planning_prompt(goal)
    assert goal in p


def test_execution_prompt_contains_step_index():
    p = make_execution_prompt("Plan", 2, ["obs1"])
    assert "étape 2" in p
