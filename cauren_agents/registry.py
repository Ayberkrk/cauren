from __future__ import annotations

from collections.abc import Callable

from cauren_agents.bridge.agent import build_bridge_agent
from cauren_agents.civil.agent import build_civil_agent

from .base import SectorAgent


class AgentRegistry:
    def __init__(self, agents: list[SectorAgent]):
        self._agents = {agent.schema.agent_id: agent for agent in agents}

    def get(self, agent_id: str) -> SectorAgent:
        try:
            return self._agents[agent_id]
        except KeyError as exc:
            raise ValueError(f"Unknown Cauren agent: {agent_id}") from exc

    def all(self) -> list[SectorAgent]:
        return list(self._agents.values())

    def ids(self) -> list[str]:
        return sorted(self._agents)


_AGENT_BUILDERS: dict[str, Callable[[], SectorAgent]] = {
    "cauren-civil": build_civil_agent,
    # Registered so training tools can target it directly
    # (--agents cauren-bridge, data/cauren_bridge/) -- deliberately left
    # out of DEFAULT_AGENT_IDS below so it does not appear on the live
    # API/router until it has an actual trained checkpoint behind it.
    "cauren-bridge": build_bridge_agent,
}

DEFAULT_AGENT_IDS: tuple[str, ...] = ("cauren-civil",)


def build_registry(agent_ids: list[str] | tuple[str, ...] | None = None) -> AgentRegistry:
    requested = list(agent_ids) if agent_ids is not None else sorted(_AGENT_BUILDERS)
    missing = [agent_id for agent_id in requested if agent_id not in _AGENT_BUILDERS]
    if missing:
        raise ValueError(f"Unknown Cauren agents requested: {', '.join(sorted(missing))}")
    return AgentRegistry([_AGENT_BUILDERS[agent_id]() for agent_id in requested])


def build_default_registry() -> AgentRegistry:
    return build_registry(DEFAULT_AGENT_IDS)
