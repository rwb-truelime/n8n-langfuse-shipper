"""Input/output normalization and LangChain system prompt stripping.

This module unwraps nested structures commonly emitted by n8n nodes, flattening
AI channel wrappers and generic JSON containers while preserving binary blocks.
Also strips system prompts from LangChain LMChat node inputs using exact marker
detection.

Unwrapping Functions:
    unwrap_ai_channel: Flatten single-key ai_* dicts containing json arrays
    unwrap_generic_json: Extract json fields from nested list/dict structures
    normalize_node_io: Apply both unwrappers and merge back binary blocks

LLM Parameter Extraction:
    extract_generation_input_and_params: Separate messages from LLM config params

System Prompt Stripping:
    strip_system_prompt_from_langchain_lmchat: Remove System segment using
    case-insensitive "human:" marker

AI Channel Structure:
    Input: {"ai_languageModel": [[{"json": {...}}]]}
    Output: {...} (unwrapped json content)

Generic JSON Structure:
    Input: [{"json": {...}}, {"json": {...}}]
    Output: [{...}, {...}] (list of unwrapped objects)

Binary Block Preservation:
    When unwrapping would lose top-level binary dict, merges it back into
    normalized output to prevent media placeholder loss.

Recursive Search Pattern Philosophy:
    Multiple functions in this module use recursive depth-first traversal through
    nested dict/list structures. These are kept SEPARATE (not abstracted) because:

    1. Different Semantics: Some are finders (read-only, early exit), others are
       mutators (deep copy, full traversal)
    2. Different Goals: Finding specific keys vs transforming content vs collecting
       assets
    3. Different Performance: Early exit optimization vs full structure traversal
    4. Different Depth Limits: 5 vs 25 depending on expected n8n nesting patterns
    5. Clarity over DRY: Self-contained functions are easier to understand than
       complex generic helpers with callbacks and predicates

    Each recursive function documents its specific traversal characteristics and
    contrasts itself with related functions to clarify when to use which pattern.

Design Notes:
    - Deduplication using JSON signature prevents redundant unwrapped objects
    - System prompt split occurs BEFORE normalization to preserve structure
    - Recursive search bounded by depth limits (5-25) and item counts (100-150)
    - Fail-open: returns original input on any error
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

__all__ = [
    "unwrap_ai_channel",
    "unwrap_generic_json",
    "normalize_node_io",
    "strip_system_prompt_from_langchain_lmchat",
    "extract_generation_input_and_params",
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


def extract_generation_input_and_params(
    input_obj: Any,
    is_generation: bool
) -> tuple[Any, Dict[str, Any]]:
    """Extract clean input and LLM parameters for generation spans.

    For generation nodes with a `messages` array in input, extracts:
    - Clean input: content string(s) from messages array
        * Single message: returns the content string directly
        * Multiple messages: returns list of content strings
    - LLM params: all other keys (max_tokens, temperature, etc.) as metadata

    This provides clean prompt visibility in Langfuse UI while preserving
    full LLM configuration in metadata for reproducibility.

    Recursively searches nested structures (up to depth 5) for dicts
    containing `messages` array, similar to system prompt stripping.

    Args:
        input_obj: Raw input object (may contain messages + config params)
        is_generation: Whether span is classified as generation

    Returns:
        Tuple of (clean_input, llm_params_metadata)
        - clean_input: content string (single) or list of strings (multiple),
          else original
        - llm_params_metadata: dict with n8n.llm.* keys for config params
    """
    llm_params: Dict[str, Any] = {}

    if not is_generation:
        return input_obj, llm_params

    def _find_messages_dict(obj: Any, depth: int = 0) -> tuple[
        Optional[Dict[str, Any]], int
    ]:
        """Recursively search for dict containing messages array.

        Traversal Pattern:
            - Depth-first search through nested dict/list structures
            - Early exit optimization: returns immediately on first match
            - Read-only operation: does not modify input data
            - Bounded recursion: stops at MAX_DEPTH to prevent infinite loops

        Use Case:
            Find the first dict containing a "messages" array (non-empty list)
            to extract LLM parameters. Production n8n data nests messages at
            depth 4 inside ai_languageModel wrappers.

        Contrast with _process_messages_recursive:
            - This function FINDS (read-only, early exit)
            - That function MUTATES (deep copy, full traversal)
            - Different depth limits (5 vs 25)
            - Different return types (tuple vs bool)

        Returns:
            Tuple of (dict_with_messages, depth_found) or (None, -1)
        """
        MAX_DEPTH = 5
        if depth > MAX_DEPTH:
            return None, -1

        if isinstance(obj, dict):
            # Check if current dict has messages array
            messages = obj.get("messages")
            if isinstance(messages, list) and len(messages) > 0:
                logger.debug(
                    "Found messages dict at depth %d: keys=%s, "
                    "messages_count=%d",
                    depth,
                    list(obj.keys())[:20],
                    len(messages),
                )
                return obj, depth

            # Recursively search nested structures
            for value in obj.values():
                result, found_depth = _find_messages_dict(value, depth + 1)
                if result is not None:
                    return result, found_depth

        elif isinstance(obj, list):
            # Search list items
            for item in obj:
                result, found_depth = _find_messages_dict(item, depth + 1)
                if result is not None:
                    return result, found_depth

        return None, -1

    # Search for dict containing messages array
    messages_dict, depth_found = _find_messages_dict(input_obj)

    if messages_dict is None:
        logger.debug(
            "No messages array found in input_obj (searched up to depth 5)"
        )
        return input_obj, llm_params

    # Extract messages array
    messages = messages_dict["messages"]

    # Extract content strings from messages as clean input
    if messages and len(messages) > 0:
        # Extract content from each message
        content_list = []
        for msg in messages:
            if isinstance(msg, dict) and "content" in msg:
                content_list.append(msg["content"])
            elif isinstance(msg, str):
                content_list.append(msg)
            else:
                # Fallback: include the message as-is
                content_list.append(msg)

        # If single message, return just the string; if multiple, return array
        if len(content_list) == 1:
            clean_input = content_list[0]
        else:
            clean_input = content_list
    else:
        clean_input = messages

    # Extract all other keys as LLM parameters
    param_keys = []
    for key, value in messages_dict.items():
        if key == "messages":
            continue

        # Special handling for 'options' dict - flatten it
        if key == "options" and isinstance(value, dict):
            for opt_key, opt_value in value.items():
                try:
                    import json
                    json.dumps(opt_value)
                    llm_params[f"n8n.llm.{opt_key}"] = opt_value
                    param_keys.append(opt_key)
                except (TypeError, ValueError):
                    llm_params[f"n8n.llm.{opt_key}"] = str(opt_value)
                    param_keys.append(opt_key)
            continue

        # Store under n8n.llm.* prefix in metadata
        try:
            # Convert to JSON-serializable format
            import json
            # Test serializability
            json.dumps(value)
            llm_params[f"n8n.llm.{key}"] = value
            param_keys.append(key)
        except (TypeError, ValueError):
            # Non-serializable values → convert to string
            llm_params[f"n8n.llm.{key}"] = str(value)
            param_keys.append(key)

    logger.debug(
        "Extracted %d LLM parameters at depth %d: %s; "
        "clean input type=%s (from %d messages)",
        len(llm_params),
        depth_found,
        ", ".join(param_keys[:10]),  # Show first 10 keys
        type(clean_input).__name__,
        len(messages),
    )

    return clean_input, llm_params


def strip_system_prompt_from_langchain_lmchat(input_obj: Any, node_type: str) -> Any:
    """Strip System prompts from LangChain LMChat messages.

    Searches for 'human:' (case-insensitive) as the consistent split marker
    across all message formats. Strips everything before AND INCLUDING the
    'human:' marker and any following whitespace.

    LangChain LMChat Message Formats (INPUT):
        - "System: ...\n\n## START PROCESSING\n\nHuman: ## ..."
        - "System: ...\n\nHuman: ..." (no ## markers)
        - "System: ...\nhuman: ..." (lowercase)
        - "System: ...\nHUMAN: ..." (uppercase)

    Stripping Behavior (OUTPUT):
        - "System: foo\n\nHuman: ## Order" → "## Order"
        - "System: bar\nhuman:  test" → "test"
        - Message without "human:" → unchanged

    Only consistent marker across all formats is "human:" (case-insensitive).
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

        def _find_human_marker(text: str) -> int:
            """Find first occurrence of 'human:' (case-insensitive).

            Returns the index AFTER 'human:' and any following whitespace,
            or -1 if not found.

            Example:
                "System: foo\n\nHuman: ## Order" -> returns index pointing to "##"
                "human:  test" -> returns index pointing to "test"
            """
            text_lower = text.lower()
            idx = text_lower.find("human:")
            if idx == -1:
                return -1

            # Skip past "human:" (6 characters)
            idx += 6

            # Skip any following whitespace (but NOT newlines with content)
            # We want to preserve "## " as it's markdown header syntax
            while idx < len(text) and text[idx] in (' ', '\t'):
                idx += 1

            return idx

        def _process_messages_recursive(obj: Any, depth: int = 0) -> bool:
            """Recursively find and mutate all messages arrays.

            Traversal Pattern:
                - Depth-first search through nested dict/list structures
                - Full traversal: processes ALL messages (no early exit)
                - Mutation operation: modifies deep copy in-place
                - Bounded recursion: stops at depth 25, limits list items to 100

            Use Case:
                Find ALL "messages" arrays within deeply nested structures and
                strip system prompts from each message string. Must traverse
                entire structure since multiple messages may exist.

            Contrast with _find_messages_dict:
                - This function MUTATES (deep copy, full traversal)
                - That function FINDS (read-only, early exit)
                - Different depth limits (25 vs 5)
                - Different return types (bool vs tuple)

            Returns:
                Boolean indicating whether any modifications were made
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
                                if isinstance(value, str):
                                    split_idx = _find_human_marker(value)
                                    if split_idx != -1:
                                        msg[key] = value[split_idx:]
                                        modified_any = True
                                        logger.debug(
                                            "Stripped system prompt (dict depth=%s); node_type=%s removed_chars=%s",
                                            depth,
                                            node_type,
                                            split_idx,
                                        )
                        elif isinstance(msg, str):
                            split_idx = _find_human_marker(msg)
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
