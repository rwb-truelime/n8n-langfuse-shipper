"""Model extraction helpers from legacy mapper.

Pure functions; no side effects besides deterministic logging.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

from ..models.n8n import WorkflowNode

logger = logging.getLogger(__name__)

__all__ = ["extract_model_value", "looks_like_model_param_key", "extract_model_from_parameters"]


def extract_model_value(data: Any) -> Optional[str]:
    if not data:
        return None
    keys = ("model", "model_name", "modelId", "model_id")
    try:
        if isinstance(data, dict):
            for chan_key, chan_val in data.items():
                if isinstance(chan_key, str) and chan_key.startswith("ai_") and isinstance(chan_val, list):
                    for outer in chan_val:
                        if not isinstance(outer, list):
                            continue
                        for item in outer:
                            if not isinstance(item, dict):
                                continue
                            j = item.get("json") if isinstance(item.get("json"), dict) else None
                            if not j:
                                continue
                            for k in keys:
                                val = j.get(k)
                                if isinstance(val, str) and val:
                                    return val
                            opts = j.get("options") if isinstance(j.get("options"), dict) else None
                            if opts:
                                for k in keys:
                                    val = opts.get(k)
                                    if isinstance(val, str) and val:
                                        return val
    except Exception:
        pass

    def _direct_scan(obj: Dict[str, Any]) -> Optional[str]:
        for k in keys:
            v = obj.get(k)
            if isinstance(v, str) and v:
                return v
        opts = obj.get("options") if isinstance(obj.get("options"), dict) else None
        if opts:
            for k in keys:
                v = opts.get(k)
                if isinstance(v, str) and v:
                    return v
        return None

    if isinstance(data, dict):
        direct = _direct_scan(data)
        if direct:
            return direct
        j = data.get("json") if isinstance(data.get("json"), dict) else None
        if j:
            direct = _direct_scan(j)
            if direct:
                return direct

    queue: Deque[Tuple[Any, int]] = deque([(data, 0)])
    visited = 0
    max_nodes = 800
    max_depth = 25
    while queue and visited < max_nodes:
        current, depth = queue.popleft()
        visited += 1
        try:
            if isinstance(current, dict):
                direct = _direct_scan(current)
                if direct:
                    return direct
                if depth < max_depth:
                    for v in current.values():
                        queue.append((v, depth + 1))
            elif isinstance(current, list):
                if depth < max_depth:
                    for item in current[:150]:
                        queue.append((item, depth + 1))
            elif isinstance(current, str):
                s = current.strip()
                if 10 < len(s) < 50_000 and s.startswith('{') and 'model' in s:
                    import json
                    try:
                        parsed = json.loads(s)
                        if isinstance(parsed, dict):
                            queue.append((parsed, depth + 1))
                    except Exception:
                        pass
        except Exception:
            continue
    return None


def looks_like_model_param_key(key: str) -> bool:
    lk = key.lower()
    if lk == "mode":
        return False
    if lk in {"modelprovider", "model_type", "modeltype", "modelprovidername"}:
        return False
    return ("model" in lk) or ("deployment" in lk)


def extract_model_from_parameters(node: WorkflowNode) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    params = getattr(node, "parameters", None)
    if not isinstance(params, dict):
        return None
    from collections import deque as _dq
    queue = _dq([(params, "parameters", 0)])
    seen = 0
    max_nodes = 120
    priority_keys = [
        "model",
        "customModel",
        "deploymentName",
        "deployment",
        "modelName",
        "model_id",
        "modelId",
    ]
    result: Optional[Tuple[str, str, Dict[str, Any]]] = None
    azure_flag = "azure" in node.type.lower()
    while queue and seen < max_nodes:
        current, path, depth = queue.popleft()
        seen += 1
        if result is not None:
            break
        for pk in priority_keys:
            if pk in current:
                v = current.get(pk)
                if isinstance(v, str) and v and not v.startswith("={{"):
                    meta_pk: Dict[str, Any] = {}
                    if azure_flag:
                        meta_pk["n8n.model.is_deployment"] = True
                    result = (v, f"{path}.{pk}", meta_pk)
                    break
        for k, v in current.items():
            if (
                isinstance(v, str)
                and v
                and not v.startswith("={{")
                and looks_like_model_param_key(k)
            ):
                meta_k: Dict[str, Any] = {}
                if azure_flag:
                    meta_k["n8n.model.is_deployment"] = True
                result = (v, f"{path}.{k}", meta_k)
                break
        if depth < 25:
            for k, v in current.items():
                if isinstance(v, dict):
                    queue.append((v, f"{path}.{k}", depth + 1))
    return result
