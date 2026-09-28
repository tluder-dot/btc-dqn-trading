"""Freeze the final model: EXP-12 as an ensemble of 20 networks (seeds 0 to 19).

Writes models/final/:
    ensemble.pt       weights and config of the 20 networks (no optimizer state)
    config.json       the experiment config (features, positions, gamma, ...)
    normalizer.json   z-score statistics fitted on dev-train (applied to val and eval)

Usage (from the project root): python scripts/freeze_final_model.py
"""

import json
import shutil
import sys
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from rl_trading.agents import save_ensemble  # noqa: E402

EXP_ID = "exp12_mr_turnover_penalty"
SEEDS = list(range(20))  # fixed before seeds 12 to 19 were trained, no selection among seeds

run_root = ROOT / "runs" / EXP_ID
out_dir = ROOT / "models" / "final"
out_dir.mkdir(parents=True, exist_ok=True)

checkpoints = [run_root / f"seed_{s}" / "last.pt" for s in SEEDS]
missing = [str(p) for p in checkpoints if not p.exists()]
if missing:
    raise FileNotFoundError(f"missing checkpoints: {missing}")

# Every seed was fitted on the same dev-train rows, so the statistics must be identical
normalizers = []
for s in SEEDS:
    with open(run_root / f"seed_{s}" / "normalizer.json") as f:
        normalizers.append(json.load(f))
assert all(n == normalizers[0] for n in normalizers), "normalizer statistics differ between seeds"

save_ensemble(checkpoints, out_dir / "ensemble.pt", metadata={
    "exp_id": EXP_ID, "seeds": SEEDS, "model": "last", "frozen_on": str(date.today()),
    "normalizer": normalizers[0],
})
shutil.copy(ROOT / "configs" / f"{EXP_ID}.json", out_dir / "config.json")
with open(out_dir / "normalizer.json", "w") as f:
    json.dump(normalizers[0], f, indent=2)
print(f"frozen {len(SEEDS)} networks of {EXP_ID} into {out_dir}")
