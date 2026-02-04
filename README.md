# simple-ai-agent

Agent IA minimal (plan -> exécution étape par étape -> réponse finale).
Mode mock disponible pour développer sans crédits.

## Setup (Windows Git Bash)
```bash
python -m venv .venv
source .venv/Scripts/activate
pip install -U pip
pip install -e ".[dev]"
cp .env.example .env
