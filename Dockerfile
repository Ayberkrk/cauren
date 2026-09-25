FROM python:3.12-slim

WORKDIR /app

COPY pyproject.toml README.md LICENSE ./
COPY api ./api
COPY cauren_agents ./cauren_agents
COPY cauren_core ./cauren_core
COPY cauren_physics ./cauren_physics
COPY tools ./tools

RUN pip install --no-cache-dir ".[api]"

EXPOSE 8000

CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
