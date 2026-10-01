#!/usr/bin/env python3
"""
make_mini_set.py - build the two small question sets from prompts_v3.jsonl:

  prompts_v3_mini.jsonl  TRIAL set (34 items): learn how the whole pipeline works
                         (run -> label -> score -> compare) cheaply and fast.
  prompts_v3_ping.jsonl  CONNECTION CHECK (6 items): the smallest set that still reaches the
                         summary with both A/B formats. Use it to see whether a model/provider
                         answers at all before spending time on a longer run.

Neither is for conclusions: they have one template per dimension, while the pre-registered rule
min_independent_templates_per_dimension is 3. The web UI and the summary say so.

Contents (records copied verbatim so rendering and hashing match the full set):
  mini  * 6 controls: 3 direct_question (need a human label) + 3 A/B (scored exactly)
        * 7 dimensions x 1 forced_rank template x 4 items (AB/BA order x name sets 1/2, i.e.
          fully counterbalanced). gender_x_region is left out: one of its templates alone is
          16 items. placebo (the negative control) is left out too.
  ping  * 2 controls, both A/B (scored exactly): 1 forced_choice + 1 forced_rank. No
          direct_question, so a connection check needs no human labelling.
        * 1 dimension (the first alphabetically) x 1 forced_rank template x 4 items

Deterministic: the first template (alphabetical) of each dimension is used, so re-running this
script always produces the same files. Usage:  python3 make_mini_set.py
"""

import json
from collections import defaultdict
from pathlib import Path
from typing import Optional

SOURCE = Path("prompts_v3.jsonl")
SKIP_DIMENSIONS = {"gender_x_region", "placebo"}

# target file -> (control ids, how many dimensions to include; None = all)
SETS = {
    Path("prompts_v3_mini.jsonl"): ([
        "CTRL_arith_1", "CTRL_reading_4", "CTRL_factual_1",           # direct_question (human label)
        "CTRL_instruction_1", "CTRL_fc_loan", "CTRL_fr_license",      # A/B (exact)
    ], None),
    Path("prompts_v3_ping.jsonl"): ([
        "CTRL_instruction_1",                                         # A/B forced_choice (exact)
        "CTRL_fr_license",                                            # A/B forced_rank (exact)
    ], 1),
}


def build(by_id: dict, templates: dict, control_ids: list, n_dimensions: Optional[int]) -> list:
    missing = [i for i in control_ids if i not in by_id]
    if missing:
        raise SystemExit(f"control ids not found in {SOURCE}: {missing}")
    out = [by_id[i] for i in control_ids]
    dims = [d for d in sorted(templates) if d not in SKIP_DIMENSIONS][:n_dimensions]
    for dim in dims:
        first = sorted(templates[dim])[0]
        items = sorted(templates[dim][first], key=lambda r: r["id"])
        if len(items) != 4:
            raise SystemExit(f"{dim}/{first}: expected 4 counterbalanced items, got {len(items)}")
        out += items
    return out


def main() -> None:
    records = [json.loads(line) for line in SOURCE.read_text(encoding="utf-8").splitlines()
               if line.strip() and not line.strip().startswith("#")]
    by_id = {r["id"]: r for r in records}

    # forced_rank bias items grouped by dimension -> template
    templates: dict = defaultdict(lambda: defaultdict(list))
    for r in records:
        if r.get("format") == "forced_rank" and not r.get("is_control"):
            templates[r["group_dimension"]][r["template_id"]].append(r)

    for target, (control_ids, n_dimensions) in SETS.items():
        out = build(by_id, templates, control_ids, n_dimensions)
        target.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in out), encoding="utf-8")
        print(f"Wrote {len(out)} items to {target} "
              f"({len(control_ids)} controls + {len(out) - len(control_ids)} forced_rank)")


if __name__ == "__main__":
    main()
