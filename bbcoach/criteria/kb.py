from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml


KB_VERSION = "v1"


def _expect_keys(obj: Dict[str, Any], required: Tuple[str, ...], path: str) -> None:
    missing = [k for k in required if k not in obj]
    extra = [k for k in obj.keys() if k not in required]
    if missing:
        raise ValueError(f"Missing keys at {path}: {missing}")
    if extra:
        raise ValueError(f"Unexpected keys at {path}: {extra}")


def _expect_type(value: Any, expected_type: type, path: str) -> None:
    if not isinstance(value, expected_type):
        raise ValueError(f"Expected {expected_type.__name__} at {path}, got {type(value).__name__}")


def _validate_citation_id(citations: Dict[str, Dict[str, str]], cid: str, path: str) -> None:
    if cid not in citations:
        raise ValueError(f"Unknown citation id '{cid}' at {path}")


def validate_kb(data: Dict[str, Any]) -> Dict[str, Any]:
    _expect_keys(data, ("kb_version", "citations", "federations"), "kb")
    _expect_type(data["kb_version"], str, "kb.kb_version")
    _expect_type(data["citations"], dict, "kb.citations")
    _expect_type(data["federations"], list, "kb.federations")

    citations: Dict[str, Dict[str, str]] = {}
    for cid, cdata in data["citations"].items():
        if not isinstance(cid, str):
            raise ValueError("Citation id must be a string")
        if not isinstance(cdata, dict):
            raise ValueError(f"Citation {cid} must be a dict")
        _expect_keys(cdata, ("title", "url", "document"), f"citations.{cid}")
        for key in ("title", "url", "document"):
            _expect_type(cdata[key], str, f"citations.{cid}.{key}")
        citations[cid] = cdata

    for f_idx, fed in enumerate(data["federations"]):
        if not isinstance(fed, dict):
            raise ValueError(f"Federation entry {f_idx} must be a dict")
        _expect_keys(fed, ("id", "name", "version", "categories"), f"federations[{f_idx}]")
        for key in ("id", "name", "version"):
            _expect_type(fed[key], str, f"federations[{f_idx}].{key}")
        _expect_type(fed["categories"], list, f"federations[{f_idx}].categories")

        for c_idx, cat in enumerate(fed["categories"]):
            if not isinstance(cat, dict):
                raise ValueError(f"Category entry {f_idx}.{c_idx} must be a dict")
            _expect_keys(
                cat,
                (
                    "id",
                    "name",
                    "official_axes_text",
                    "judging_axes",
                    "source_of_truth",
                    "scoring_model",
                    "key_proxies",
                    "measurable_proxies",
                    "thresholds",
                    "feedback_rules",
                    "citations",
                ),
                f"federations[{f_idx}].categories[{c_idx}]",
            )
            _expect_type(cat["id"], str, f"categories[{c_idx}].id")
            _expect_type(cat["name"], str, f"categories[{c_idx}].name")
            _expect_type(cat["official_axes_text"], str, f"categories[{c_idx}].official_axes_text")
            _expect_type(cat["judging_axes"], list, f"categories[{c_idx}].judging_axes")
            _expect_type(cat["key_proxies"], list, f"categories[{c_idx}].key_proxies")
            _expect_type(cat["measurable_proxies"], list, f"categories[{c_idx}].measurable_proxies")
            _expect_type(cat["thresholds"], list, f"categories[{c_idx}].thresholds")
            _expect_type(cat["feedback_rules"], list, f"categories[{c_idx}].feedback_rules")
            _expect_type(cat["citations"], list, f"categories[{c_idx}].citations")

            src = cat["source_of_truth"]
            _expect_type(src, dict, f"categories[{c_idx}].source_of_truth")
            _expect_keys(src, ("title", "url", "document", "quote_hint"), f"categories[{c_idx}].source_of_truth")
            for key in ("title", "url", "document", "quote_hint"):
                _expect_type(src[key], str, f"categories[{c_idx}].source_of_truth.{key}")

            scoring = cat["scoring_model"]
            _expect_type(scoring, dict, f"categories[{c_idx}].scoring_model")
            _expect_keys(scoring, ("axis_weights", "is_official_weighting", "note"), f"categories[{c_idx}].scoring_model")
            _expect_type(scoring["axis_weights"], dict, f"categories[{c_idx}].scoring_model.axis_weights")
            _expect_type(scoring["is_official_weighting"], bool, f"categories[{c_idx}].scoring_model.is_official_weighting")
            _expect_type(scoring["note"], str, f"categories[{c_idx}].scoring_model.note")

            # Validate axis weights
            axes = [str(a) for a in cat["judging_axes"]]
            if not axes:
                raise ValueError(f"categories[{c_idx}] must include judging_axes")
            weight_keys = list(scoring["axis_weights"].keys())
            if set(weight_keys) != set(axes):
                raise ValueError(
                    f"categories[{c_idx}] axis_weights keys must match judging_axes. "
                    f"Got {weight_keys} vs {axes}"
                )
            total_w = sum(float(v) for v in scoring["axis_weights"].values())
            if abs(total_w - 1.0) > 0.02:
                raise ValueError(f"categories[{c_idx}] axis_weights must sum to 1.0 (+/-0.02), got {total_w}")

            for mp_idx, mp in enumerate(cat["measurable_proxies"]):
                if not isinstance(mp, dict):
                    raise ValueError(f"measurable_proxies[{mp_idx}] must be a dict")
                if "proxy_id" not in mp or "description" not in mp or "contributes_to" not in mp:
                    raise ValueError(f"measurable_proxies[{mp_idx}] missing required keys")
                _expect_type(mp["proxy_id"], str, f"measurable_proxies[{mp_idx}].proxy_id")
                _expect_type(mp["description"], str, f"measurable_proxies[{mp_idx}].description")
                _expect_type(mp["contributes_to"], list, f"measurable_proxies[{mp_idx}].contributes_to")
                for ax in mp["contributes_to"]:
                    if ax not in axes:
                        raise ValueError(
                            f"measurable_proxies[{mp_idx}] contributes_to axis '{ax}' not in judging_axes"
                        )
                if "from" in mp and not isinstance(mp["from"], list):
                    raise ValueError(f"measurable_proxies[{mp_idx}].from must be a list if present")

            for cid in cat["citations"]:
                _validate_citation_id(citations, str(cid), f"categories[{c_idx}].citations")

            for r_idx, rule in enumerate(cat["feedback_rules"]):
                if not isinstance(rule, dict):
                    raise ValueError(f"feedback_rules[{r_idx}] must be a dict")
                _expect_keys(
                    rule,
                    (
                        "id",
                        "when",
                        "message",
                        "action",
                        "training_prescription",
                        "citations",
                    ),
                    f"feedback_rules[{r_idx}]",
                )
                _expect_type(rule["id"], str, f"feedback_rules[{r_idx}].id")
                _expect_type(rule["when"], str, f"feedback_rules[{r_idx}].when")
                _expect_type(rule["message"], str, f"feedback_rules[{r_idx}].message")
                _expect_type(rule["action"], dict, f"feedback_rules[{r_idx}].action")
                _expect_type(rule["training_prescription"], dict, f"feedback_rules[{r_idx}].training_prescription")
                _expect_type(rule["citations"], list, f"feedback_rules[{r_idx}].citations")

                action = rule["action"]
                _expect_keys(action, ("training", "posing", "scan_corrections"), f"feedback_rules[{r_idx}].action")

                tp = rule["training_prescription"]
                _expect_keys(
                    tp,
                    ("muscle_groups", "evidence", "guidelines", "exercise_menu", "progression", "deload_notes"),
                    f"feedback_rules[{r_idx}].training_prescription",
                )

                if not rule["citations"]:
                    raise ValueError(f"feedback_rules[{r_idx}] must include citations")
                for cid in rule["citations"]:
                    _validate_citation_id(citations, str(cid), f"feedback_rules[{r_idx}].citations")

    return data


def load_kb(path: Optional[Path] = None) -> Dict[str, Any]:
    kb_path = path or (Path(__file__).resolve().parent.parent.parent / "data" / "federations" / "criteria_kb_v1.yaml")
    if not kb_path.exists():
        raise FileNotFoundError(f"Criteria KB not found: {kb_path}")
    data = yaml.safe_load(kb_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("KB root must be a mapping")
    return validate_kb(data)


def get_category(kb: Dict[str, Any], federation_id: str, category_id: str) -> Dict[str, Any]:
    for fed in kb.get("federations", []):
        if str(fed.get("id", "")).lower() != federation_id.lower():
            continue
        for cat in fed.get("categories", []):
            if str(cat.get("id", "")).lower() == category_id.lower():
                return cat
    raise KeyError(f"Category not found: {federation_id}/{category_id}")
