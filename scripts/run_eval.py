from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from baselines.oracle import run_oracle

TASK_IDS = ("refund_exception", "invoice_plus_kyc", "queue_triage", "ap_payment_run")


def main() -> None:
    for task_id in TASK_IDS:
        print(json.dumps({"task_id": task_id, "oracle": run_oracle(task_id)}, indent=2))


if __name__ == "__main__":
    main()
