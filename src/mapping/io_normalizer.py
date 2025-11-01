"""Input/output normalization and LangChain system prompt stripping.

This module unwraps nested structures commonly emitted by n8n nodes, flattening
AI channel wrappers and generic JSON containers while preserving binary blocks.
Also strips system prompts from LangChain LMChat node inputs using exact marker
detection.

Unwrapping Functions:
    unwrap_ai_channel: Flatten single-key ai_* dicts containing json arrays
    unwrap_generic_json: Extract json fields from nested list/dict structures
    normalize_node_io: Apply both unwrappers and merge back binary blocks

System Prompt Stripping:
    strip_system_prompt_from_langchain_lmchat: Remove System segment using literal
    split marker `\\n\\n## START PROCESSING\\n\\nHuman: ##`

AI Channel Structure:
    Input: {"ai_languageModel": [[{"json": {...}}]]}
    Output: {...} (unwrapped json content)

Generic JSON Structure:
    Input: [{"json": {...}}, {"json": {...}}]
    Output: [{...}, {...}] (list of unwrapped objects)

Binary Block Preservation:
    When unwrapping would lose top-level binary dict, merges it back into
    normalized output to prevent media placeholder loss.

Design Notes:
    - Deduplication using JSON signature prevents redundant unwrapped objects
    - System prompt split occurs BEFORE normalization to preserve structure
    - Recursive search bounded by depth limits (25) and item counts (100-150)
    - Fail-open: returns original input on any error
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List

logger = logging.getLogger(__name__)

__all__ = [
    "unwrap_ai_channel",
    "unwrap_generic_json",
    "normalize_node_io",
    "strip_system_prompt_from_langchain_lmchat",
]


def unwrap_ai_channel(container: Any) -> Any:
    """Unwrap single-key ai_* wrapper dicts extracting nested json objects.

    n8n AI nodes often wrap outputs as {"ai_languageModel": [[{"json": {...}}]]}.
    This function flattens such structures by recursively extracting json blocks
    or dicts containing model/tokenUsage keys.

    Returns original container when:
    - Not a single-key dict
    - Key doesn't start with "ai_"
    - No json blocks or model-related dicts found

    Deduplicates extracted objects using JSON signature to prevent redundancy.

    Args:
        container: Potential AI channel wrapper structure

    Returns:
        Unwrapped json object(s) or original container
        - Single object if only one extracted
        - List of unique objects if multiple found
        - Original container if unwrapping not applicable
    """
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
        """Recursively collect json-wrapped objects or model-related dicts.

        Args:
            v: Value to traverse (dict, list, or other).
            depth: Current recursion depth (max 25 to prevent stack overflow).

        Note:
            Modifies `collected` list in enclosing scope by appending
            discovered objects that contain model or usage keys.
        """
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
    """Extract nested json field values from list/dict wrapper structures.

    Recursively searches for {"json": {...}} patterns and extracts the json values.
    Common in n8n node outputs where data is wrapped for processing.

    Bounded by depth=25 and max 150 collected objects to prevent excessive traversal.
    Deduplicates using JSON signature.

    Args:
        container: Data structure potentially containing json wrappers

    Returns:
        Unwrapped json object(s) or original container
        - Single object if only one found
        - List of unique objects if multiple found
        - Original container if no json fields found
    """
    if not isinstance(container, dict):
        return container
    collected: List[Dict[str, Any]] = []

    def _walk(o: Any, depth: int = 0) -> None:
        """Recursively collect json-wrapped objects from nested structure.

        Args:
            o: Object to traverse (dict, list, or other).
            depth: Current recursion depth (max 25, collection cap 150).

        Note:
            Modifies `collected` list in enclosing scope by appending
            discovered json-wrapped dicts.
        """
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
    """Apply unwrapping pipeline to node I/O and preserve binary blocks.

    Orchestrates normalization process:
    1. Apply AI channel unwrapping
    2. Apply generic json unwrapping
    3. Merge back top-level binary dict if lost during unwrapping

    Binary preservation prevents media placeholder loss when output structure
    flattened.

    Args:
        obj: Raw node input or output data

    Returns:
        Tuple of (normalized_object, flags_dict)
        - normalized_object: Unwrapped data structure
        - flags_dict: Transformation flags (unwrapped_ai_channel, unwrapped_json_root)
    """
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
    """Remove system prompt segment from LangChain LMChat node inputs.

    LangChain LMChat nodes combine System and User prompts in one message. This
    function strips the System segment using exact literal marker detection.

    Split marker (literal sequence):
    \\n\\n## START PROCESSING\\n\\nHuman: ##

    Stripping logic:
    - Recursively searches for "messages" arrays up to depth 25
    - Handles both list of strings and list of dicts with content keys
    - Removes everything before "Human: ##" when marker found
    - Fail-open: returns original input on any error

    Only processes nodes with "lmchat" in type (case-insensitive).

    Args:
        input_obj: Node input data potentially containing messages
        node_type: Node type string for lmchat detection

    Returns:
        Modified input with system prompts stripped, or original on error/no match
    """
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
            """Recursively find messages arrays and strip system prompts.

            Args:
                obj: Object to traverse (dict, list, or other).
                depth: Current recursion depth (max 25 to prevent overflow).

            Returns:
                True if processing should continue to deeper levels, False
                if depth limit reached.

            Note:
                Mutates `modified` in enclosing scope by stripping system
                prompt prefix from message strings that contain the
                LangChain LMChat split marker sequence.
            """
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
