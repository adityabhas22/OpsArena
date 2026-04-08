from __future__ import annotations

import os

from openenv.core.env_server.http_server import create_app

try:
    from opsarena.models import OpsArenaObservation, RawOpsAction
    from server.environment import OpsArenaEnvironment
except ModuleNotFoundError:
    from models import OpsArenaObservation, RawOpsAction
    from server.environment import OpsArenaEnvironment


app = create_app(
    OpsArenaEnvironment,
    RawOpsAction,
    OpsArenaObservation,
    env_name="opsarena",
    max_concurrent_envs=int(os.getenv("OPSARENA_MAX_CONCURRENT_ENVS", "64") or "64"),
)


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()
