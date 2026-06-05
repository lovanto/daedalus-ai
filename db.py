from __future__ import annotations

import asyncpg

from config import settings

_pool: asyncpg.Pool | None = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(settings.DATABASE_URL)
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool:
        await _pool.close()
        _pool = None


async def fetch_agent_definition(agent_id: str) -> dict | None:
    """Return the latest definition for an agent, or None if not found."""
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        SELECT goals, intended_behaviors, constraints, success_metrics,
               unsafe_zones, confidence_threshold, sops
        FROM agent_definitions
        WHERE agent_id = $1
        ORDER BY version DESC
        LIMIT 1
        """,
        agent_id,
    )
    if row is None:
        return None
    return dict(row)
