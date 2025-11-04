"""AI model name extraction from runtime data and static workflow parameters.

This module performs multi-pass searches for model identifiers in node execution
data, with fallback to static workflow definition parameters when runtime data
lacks explicit model information.

Search Strategy:
    1. AI channel unwrapping: Look for model in ai_* channel json blocks
    2. Direct scan: Check for model/model_name/modelId/model_id at top level
    3. Breadth-first nested search: Traverse structure up to depth 25
    4. Parameter fallback: Search static workflow node parameters (generation spans)

Model Key Variants:
    - model, model_name, modelId, model_id (runtime)
    - model, customModel, deploymentName, deployment, modelName (parameters)

Azure Special Handling:
    When node type contains "azure", deployment names flagged with
    n8n.model.is_deployment=true metadata

Public Functions:
    extract_model_value: Runtime data search with breadth-first traversal
    extract_model_from_parameters: Static parameter search with priority keys
    looks_like_model_param_key: Heuristic for model-related parameter keys

Design Note:
    Bounded traversal (800 nodes, depth 25) prevents infinite loops on cyclic
    or deeply nested structures.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Any, Deque, Dict, Optional, Tuple

from ..models.n8n import WorkflowNode

logger = logging.getLogger(__name__)

__all__ = ["extract_model_value", "looks_like_model_param_key", "extract_model_from_parameters"]


def extract_model_value(data: Any) -> Optional[str]:
    """Extract model identifier from runtime node execution data.

    Performs multi-pass search strategy:
    1. AI channel unwrapping: Search ai_* keys for nested json.model fields
    2. Direct top-level scan: Check model/model_name/modelId/model_id
    3. Breadth-first traversal: Bounded search through nested structures

    Search bounded by max_nodes=800 and max_depth=25 to prevent excessive
    traversal on deeply nested or cyclic structures.

    Args:
        data: Node run output data structure

    Returns:
        Model identifier string or None if not found
    """
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
        """Scan object for model keys at top level and options subkey.

        Args:
            obj: Dictionary to scan for model key-value pairs.

        Returns:
            First non-empty string value found for any target key, or None
            if no match found.
        """
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
    """Heuristic to identify parameter keys likely containing model identifiers.

    Checks for presence of "model" or "deployment" substrings while excluding
    known false positives like "mode", "modelprovider", "model_type".

    Args:
        key: Parameter key name from workflow node definition

    Returns:
        True if key likely contains model identifier
    """
    lk = key.lower()
    if lk == "mode":
        return False
    if lk in {"modelprovider", "model_type", "modeltype", "modelprovidername"}:
        return False
    return ("model" in lk) or ("deployment" in lk)


def extract_model_from_parameters(node: WorkflowNode) -> Optional[Tuple[str, str, Dict[str, Any]]]:
    """Extract model identifier from static workflow node parameters.

    Fallback search when runtime data lacks explicit model. Searches node.parameters
    dict using priority key list and breadth-first traversal.

    Priority keys (checked first):
    - model, customModel, deploymentName, deployment, modelName, model_id, modelId

    Azure special handling:
    When node.type contains "azure", attaches n8n.model.is_deployment=true metadata
    to indicate deployment name rather than model name.

    Args:
        node: WorkflowNode with static parameters dict

    Returns:
        Tuple of (model_value, key_path, metadata_dict) or None if not found
        - model_value: Extracted model string
        - key_path: Dot-separated path to key (e.g., "parameters.model")
        - metadata_dict: Additional metadata like is_deployment flag
    """
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
