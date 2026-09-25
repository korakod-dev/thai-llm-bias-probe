#!/usr/bin/env python3
"""
make_mini_set.py - build prompts_v3_mini.jsonl, a small TRIAL question set, from prompts_v3.jsonl.

Purpose: learn how the whole pipeline works (run -> label -> score -> compare) cheaply and fast.
It is NOT for conclusions: it has one template per dimension, while the pre-registered rule
min_independent_templates_per_dimension is 3. The web UI and the summary say so.

Contents (34 items, records copied verbatim so rendering and hashing match the full set):
  * 6 controls: 3 direct_question (need a human label) + 3 A/B (scored exactly)
  * 7 dimensions x 1 forced_rank template x 4 items (AB/BA order x name sets 1/2, i.e. fully
    counterbalanced). gender_x_region is left out: one of its templates alone is 16 items.

Deterministic: the first template (alphabetical) of each dimension is used, so re-running this
script always produces the same file. Usage:  python3 make_mini_set.py
"""

import json
from collections import defaultdict
from pathlib import Path

SOURCE = Path("prompts_v3.jsonl")
TARGET = Path("prompts_v3_mini.jsonl")

CONTROL_IDS = [
    "CTRL_arith_1", "CTRL_reading_4", "CTRL_factual_1",           # direct_question (human label)
    "CTRL_instruction_1", "CTRL_fc_loan", "CTRL_fr_license",      # A/B (exact)
]
SKIP_DIMENSIONS = {"gender_x_region"}


def main() -> None:
    records = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.strip().startswith("#")]
    by_id = {r["id"]: r for r in records}
    missing = [i for i in CONTROL_IDS if i not in by_id]
    if missing:
        raise SystemExit(f"control ids not found in {SOURCE}: {missing}")

    # forced_rank bias items grouped by dimension -> template
    templates: dict = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("format") == "forced_rank" and not r.get("is_control"):
            templates[r["group_dimension"]][r["template_id"]].append(r)

    out = [by_id[i] for i in CONTROL_IDS]
    for dim in sorted(templates):
        if dim in SKIP_DIMENSIONS:
            continue
        first = sorted(templates[dim])[0]
        items = sorted(templates[dim][first], key=lambda r: r["id"])
        if len(items) != 4:
            raise SystemExit(f"{dim}/{first}: expected 4 counterbalanced items, got {len(items)}")
        out += items

    TARGET.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
    print(f"Wrote {len(out)} items to {TARGET} "
          f"({len(CONTROL_IDS)} controls + {len(out) - len(CONTROL_IDS)} forced_rank)")


if __name__ == "__main__":
    main()
