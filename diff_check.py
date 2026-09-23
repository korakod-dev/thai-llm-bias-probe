"""
diff_check.py — validate prompts_v3.jsonl before any real run.

Hard errors (exit 1):
  * pair structure   — every pair_id has exactly 2 records of the same format
  * forced_rank pair — AB/BA contain identical candidate texts and identical
                       scenario_context, just swapped
  * candidate diff   — within one forced_rank item, the two candidate texts
                       differ only in name + group signal (similarity check
                       after removing the names)
  * forced_choice    — pair scenario_context similarity >= 0.85
  * option count     — every forced_choice / forced_rank item has exactly 2 options
  * counterbalancing — each comparison has name_set 1 and 2; when
                       name_counterbalanced is true each name appears once with
                       each group_variant; when false the two name sets use
                       different names
  * gender_x_region  — each template covers all 4 cells, and every comparison
                       differs in exactly ONE factor (no diagonal pairs)
  * template count   — >= min_independent_templates_per_dimension distinct
                       forced_rank templates per dimension (config_v3_thresholds.yaml)
  * controls         — expected_answer present; A/B-format controls expect "A"/"B"

Warnings (printed, exit 0):
  * fewer than MIN_CONTROL_ITEMS controls
  * A/B expected answers on controls not roughly balanced
  * supplementary forced_choice items (listed so they aren't pooled by mistake)

Usage:
    python diff_check.py prompts_v3.jsonl
"""

import sys
import json
import difflib
from collections import defaultdict, Counter

MIN_CONTROL_ITEMS = 20          # sanity floor so the 0.80 threshold means something
CANDIDATE_SIMILARITY_MIN = 0.70  # after names are removed
FORCED_CHOICE_SIMILARITY_MIN = 0.85
DEFAULT_MIN_TEMPLATES = 3


def load_records(path):
    records = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def load_min_templates(config_path="config_v3_thresholds.yaml"):
    """Read the pre-registered value; fall back to 3 if PyYAML/config is missing."""
    try:
        import yaml
        with open(config_path, encoding="utf-8") as f:
            return int(yaml.safe_load(f)["min_independent_templates_per_dimension"])
    except Exception:
        return DEFAULT_MIN_TEMPLATES


def similarity(a, b):
    return difflib.SequenceMatcher(None, a, b).ratio()


def check_pairs(records, problems):
    pairs = defaultdict(list)
    for r in records:
        if not r.get("is_control") and r.get("pair_id"):
            pairs[r["pair_id"]].append(r)

    for pair_id, items in pairs.items():
        if len(items) != 2:
            problems.append(f"[{pair_id}] expected 2 records, found {len(items)}")
            continue
        a, b = items
        if a["format"] != b["format"]:
            problems.append(f"[{pair_id}] format mismatch: {a['format']} vs {b['format']}")
            continue

        if a["format"] == "forced_rank":
            if sorted(o["text"] for o in a["options"]) != sorted(o["text"] for o in b["options"]):
                problems.append(f"[{pair_id}] AB/BA candidate texts differ — not a pure swap")
            if a["options"][0]["text"] != b["options"][1]["text"]:
                problems.append(f"[{pair_id}] AB/BA not swapped (A of one should be B of the other)")
            if a["scenario_context"] != b["scenario_context"]:
                problems.append(f"[{pair_id}] AB/BA scenario_context differs")
            if {a.get("order_variant"), b.get("order_variant")} != {"AB", "BA"}:
                problems.append(f"[{pair_id}] order_variant should be one AB and one BA")
        elif a["format"] == "forced_choice":
            ratio = similarity(a["scenario_context"], b["scenario_context"])
            if ratio < FORCED_CHOICE_SIMILARITY_MIN:
                problems.append(
                    f"[{pair_id}] scenario_context similarity {ratio:.2f} — may differ in more than the group signal")
    return len(pairs)


def check_options(records, problems):
    for r in records:
        if r["format"] in ("forced_choice", "forced_rank"):
            if len(r.get("options") or []) != 2:
                problems.append(f"[{r['id']}] {r['format']} must have exactly 2 options (lesson 1)")


def check_candidate_diff(records, problems):
    """Two candidates in one item should read the same apart from name + signal."""
    for r in records:
        if r["format"] != "forced_rank" or r.get("is_control"):
            continue
        a, b = r["options"]
        ta = a["text"].replace(a.get("name", ""), "")
        tb = b["text"].replace(b.get("name", ""), "")
        ratio = similarity(ta, tb)
        if ratio < CANDIDATE_SIMILARITY_MIN:
            problems.append(f"[{r['id']}] candidates only {ratio:.2f} similar after removing names — "
                            f"they may differ in more than the group signal")
        if a.get("name") and a.get("name") == b.get("name"):
            problems.append(f"[{r['id']}] both candidates share the name '{a['name']}'")


def check_counterbalancing(records, problems):
    """Per comparison: name sets 1 and 2 must exist and follow the declared scheme."""
    comps = defaultdict(dict)  # comparison_id -> {name_set: {variant: name}}
    flags = {}
    for r in records:
        if r["format"] != "forced_rank" or r.get("is_control"):
            continue
        cid = r.get("comparison_id")
        if not cid:
            problems.append(f"[{r['id']}] forced_rank item missing comparison_id")
            continue
        comps[cid][r.get("name_set")] = {o["group_variant"]: o.get("name") for o in r["options"]}
        flags[cid] = r.get("name_counterbalanced")

    for cid, sets in comps.items():
        if set(sets) != {1, 2}:
            problems.append(f"[{cid}] needs name_set 1 and 2, found {sorted(sets)}")
            continue
        s1, s2 = sets[1], sets[2]
        if flags[cid]:
            # Same two names, each attached to the other variant in set 2.
            if sorted(s1.values()) != sorted(s2.values()) or any(s1[v] == s2[v] for v in s1):
                problems.append(f"[{cid}] name_counterbalanced=true but names are not swapped across sets")
        else:
            if any(s1[v] == s2[v] for v in s1):
                problems.append(f"[{cid}] name sets 1 and 2 reuse the same name for a variant")


def check_gxr(records, problems):
    by_template = defaultdict(set)
    for r in records:
        if r.get("group_dimension") != "gender_x_region" or r["format"] != "forced_rank":
            continue
        # Older drafts lack variants_compared; fall back to the option variants.
        v1, v2 = r.get("variants_compared") or [o["group_variant"] for o in r["options"]]
        by_template[r["template_id"]].update((v1, v2))
        g1, reg1 = v1.split("_")
        g2, reg2 = v2.split("_")
        if (g1 != g2) + (reg1 != reg2) != 1:
            problems.append(f"[{r['id']}] gender_x_region comparison {v1} vs {v2} "
                            f"must differ in exactly one factor")
    cells = {"female_isan", "female_bangkok", "male_isan", "male_bangkok"}
    for tid, seen in by_template.items():
        if seen != cells:
            problems.append(f"[{tid}] 2x2 incomplete — missing {sorted(cells - seen)}")


def check_template_counts(records, problems, min_templates):
    fr = defaultdict(set)
    dims = set()
    for r in records:
        if r.get("is_control"):
            continue
        dims.add(r["group_dimension"])
        if r["format"] == "forced_rank":
            fr[r["group_dimension"]].add(r["template_id"])
    for dim in sorted(dims):
        n = len(fr[dim])
        if n < min_templates:
            problems.append(f"[{dim}] only {n} forced_rank template(s) — needs >={min_templates}")
    return {d: len(t) for d, t in fr.items()}


def check_controls(records, problems, warnings):
    controls = [r for r in records if r.get("is_control")]
    if len(controls) < MIN_CONTROL_ITEMS:
        warnings.append(f"only {len(controls)} control items (target >= {MIN_CONTROL_ITEMS})")
    letters = Counter()
    for r in controls:
        exp = r.get("expected_answer")
        if exp in (None, ""):
            problems.append(f"[{r['id']}] control item has no expected_answer")
        if r["format"] in ("forced_choice", "forced_rank"):
            if exp not in ("A", "B"):
                problems.append(f"[{r['id']}] A/B control must expect 'A' or 'B', got {exp!r}")
            else:
                letters[exp] += 1
    total = letters["A"] + letters["B"]
    if total and abs(letters["A"] - letters["B"]) > max(1, total // 4):
        warnings.append(f"control A/B answers unbalanced: A={letters['A']} B={letters['B']}")
    return len(controls), letters


def main(path):
    records = load_records(path)
    problems, warnings = [], []
    min_templates = load_min_templates()

    n_pairs = check_pairs(records, problems)
    check_options(records, problems)
    check_candidate_diff(records, problems)
    check_counterbalancing(records, problems)
    check_gxr(records, problems)
    fr_counts = check_template_counts(records, problems, min_templates)
    n_ctrl, letters = check_controls(records, problems, warnings)

    supp = sorted({r["template_id"] for r in records if r.get("analysis_role") == "supplementary"})
    if supp:
        warnings.append(f"supplementary forced_choice templates (analyse separately, don't pool): {', '.join(supp)}")

    print(f"records={len(records)} pairs={n_pairs} controls={n_ctrl} (A={letters['A']} B={letters['B']})")
    print("forced_rank templates per dimension: " +
          ", ".join(f"{d}={n}" for d, n in sorted(fr_counts.items())))
    for w in warnings:
        print(f"  warning: {w}")

    if problems:
        print(f"Found {len(problems)} issue(s):")
        for p in problems:
            print(f"  - {p}")
        sys.exit(1)
    print("All checks passed.")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python diff_check.py prompts_v3.jsonl")
        sys.exit(1)
    main(sys.argv[1])
