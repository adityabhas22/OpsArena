from __future__ import annotations

import json

from baselines.oracle import run_oracle


def main() -> None:
    for task_id in ("refund_exception", "invoice_plus_kyc", "queue_triage"):
        print(json.dumps({"task_id": task_id, "oracle": run_oracle(task_id)}, indent=2))


if __name__ == "__main__":
    main()
