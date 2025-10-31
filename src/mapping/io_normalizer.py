"""I/O normalization & system prompt stripping utilities.

Extracted from mapper.py. Behavior MUST remain identical to legacy inline
implementation (tests assert parity). Pure functions only.
"""
from __future__ import annotations
from typing import Any, Dict, List
import logging

logger = logging.getLogger(__name__)

__all__ = [
    "unwrap_ai_channel",
    "unwrap_generic_json",
    "normalize_node_io",
    "strip_system_prompt_from_langchain_lmchat",
]


def unwrap_ai_channel(container: Any) -> Any:
    if not isinstance(container, dict) or len(container) != 1:
        return container
    try:
        (only_key, value) = next(iter(container.items()))
    except Exception:
        return container
    if not (isinstance(only_key, str) and only_key.startswith("ai_")):
        return container
    collected: List[Dict[str, Any]] = []

    def _walk(v: Any, depth: int = 0) -> None:
        if depth > 25:
            return
        if isinstance(v, list):
            for item in v[:100]:
                _walk(item, depth + 1)
        elif isinstance(v, dict):
            j = v.get("json") if isinstance(v.get("json"), dict) else None
            if j:
                collected.append(j)
            else:
                if any(
                    k in v
                    for k in (
                        "model",
                        "model_name",
                        "modelId",
                        "model_id",
                        "tokenUsage",
                        "tokenUsageEstimate",
                    )
                ):
                    collected.append(v)

    try:
        _walk(value)
    except Exception:
        return container
    if not collected:
        return container
    if len(collected) == 1:
        return collected[0]
    import json
    seen: set[str] = set()
    uniq: List[Dict[str, Any]] = []
    for c in collected:
        try:
            sig = json.dumps(c, sort_keys=True)[:4000]
        except Exception:
            sig = str(id(c))
        if sig not in seen:
            seen.add(sig)
            uniq.append(c)
    return uniq


def unwrap_generic_json(container: Any) -> Any:
    if not isinstance(container, dict):
        return container
    collected: List[Dict[str, Any]] = []

    def _walk(o: Any, depth: int = 0) -> None:
        if depth > 25 or len(collected) >= 150:
            return
        if isinstance(o, dict):
            j = o.get("json") if isinstance(o.get("json"), dict) else None
            if j:
                collected.append(j)
            else:
                for v in o.values():
                    _walk(v, depth + 1)
        elif isinstance(o, list):
            for item in o[:100]:
                _walk(item, depth + 1)

    try:
        _walk(container)
    except Exception:
        return container
    if not collected:
        return container
    if len(collected) == 1:
        return collected[0]
    import json
    seen: set[str] = set()
    uniq: List[Dict[str, Any]] = []
    for c in collected:
        try:
            sig = json.dumps(c, sort_keys=True)[:4000]
        except Exception:
            sig = str(id(c))
        if sig not in seen:
            seen.add(sig)
            uniq.append(c)
    return uniq


def normalize_node_io(obj: Any) -> tuple[Any, Dict[str, bool]]:
    flags: Dict[str, bool] = {}
    base = obj
    if isinstance(base, dict):
        binary_block = base.get("binary") if isinstance(base.get("binary"), dict) else None
        after_ai = unwrap_ai_channel(base)
        if after_ai is not base:
            flags["unwrapped_ai_channel"] = True
        after_json = unwrap_generic_json(after_ai)
        if after_json is not after_ai:
            flags["unwrapped_json_root"] = True
        if binary_block and isinstance(after_json, dict) and "binary" not in after_json:
            merged = dict(after_json)
            merged["binary"] = binary_block
            after_json = merged
        return after_json, flags
    return base, flags


_SPLIT_MARKER = "\n\n## START PROCESSING\n\nHuman: ##"
_USER_START = "Human: ##"


def strip_system_prompt_from_langchain_lmchat(input_obj: Any, node_type: str) -> Any:
    if not isinstance(node_type, str):
        return input_obj
    node_type_lower = node_type.lower()
    if "lmchat" not in node_type_lower:
        return input_obj
    try:
        import copy
        modified = copy.deepcopy(input_obj)
        modified_any = False

        def _process_messages_recursive(obj: Any, depth: int = 0) -> bool:
            nonlocal modified_any
            if depth > 25:
                return False
            if isinstance(obj, dict):
                if "messages" in obj and isinstance(obj["messages"], list):
                    messages = obj["messages"]
                    for i, msg in enumerate(messages):
                        if isinstance(msg, dict):
                            for key, value in list(msg.items()):
                                if isinstance(value, str) and _SPLIT_MARKER in value:
                                    split_idx = value.find(_USER_START)
                                    if split_idx != -1:
                                        msg[key] = value[split_idx:]
                                        modified_any = True
                                        logger.debug(
                                            "Stripped system prompt (dict depth=%s); node_type=%s removed_chars=%s",
                                            depth,
                                            node_type,
                                            split_idx,
                                        )
                        elif isinstance(msg, str) and _SPLIT_MARKER in msg:
                            split_idx = msg.find(_USER_START)
                            if split_idx != -1:
                                messages[i] = msg[split_idx:]
                                modified_any = True
                                logger.debug(
                                    "Stripped system prompt (str depth=%s); node_type=%s removed_chars=%s",
                                    depth,
                                    node_type,
                                    split_idx,
                                )
                for value in obj.values():
                    _process_messages_recursive(value, depth + 1)
            elif isinstance(obj, list):
                for item in obj[:100]:
                    _process_messages_recursive(item, depth + 1)
            return modified_any

        _process_messages_recursive(modified)
        return modified if modified_any else input_obj
    except Exception as e:  # pragma: no cover (defensive fail-open)
        logger.debug(
            "Failed to strip system prompt from lmChat input: %s", e, exc_info=True
        )
        return input_obj
