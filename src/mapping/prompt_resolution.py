"""Prompt resolution via ancestor chain traversal.

This module resolves prompt metadata (name, version) for generation spans by
walking backward through NodeRun source chains to find prompt-fetch ancestors.

Resolution strategies (in precedence order):
1. Direct ancestor: Closest prompt-fetch node in source chain
2. Fingerprint matching: Compare input text when multiple candidates exist
3. None: No prompt metadata if no ancestors found

Emits rich debug metadata for transparency and troubleshooting.
"""
from __future__ import annotations

import hashlib
import logging
from typing import Any, Dict, List, Optional, Set, Tuple

from pydantic import BaseModel

from src.mapping.prompt_detection import PromptMetadata

logger = logging.getLogger(__name__)


class PromptResolutionResult(BaseModel):
    """Result of prompt resolution for a generation span.

    Contains resolved metadata plus debug information about resolution process.
    """

    prompt_name: Optional[str] = None
    prompt_version: Optional[int] = None
    original_version: Optional[int] = None  # Before env-specific resolution
    resolution_method: str = "none"  # "ancestor" | "fingerprint" | "none"
    confidence: str = "none"  # "high" | "medium" | "low" | "none"
    ancestor_distance: Optional[int] = None  # Hops to prompt ancestor
    candidate_count: int = 0  # Number of candidates considered
    ambiguous: bool = False  # Multiple candidates found


def _compute_text_fingerprint(text: str) -> str:
    """Compute lightweight fingerprint of text for comparison.

    Uses first 200 + last 200 characters plus length to create compact hash.
    Avoids expensive full-text hashing of 30K+ character prompts.

    Args:
        text: Input text (prompt or agent input)

    Returns:
        Hex hash string
    """
    if not text or len(text) < 400:
        # Short text: hash entire content
        return hashlib.sha256(text.encode("utf-8", errors="ignore")).hexdigest()

    # Long text: fingerprint strategy
    prefix = text[:200]
    suffix = text[-200:]
    length = str(len(text))

    fingerprint_input = f"{prefix}||{suffix}||{length}"
    return hashlib.sha256(
        fingerprint_input.encode("utf-8", errors="ignore")
    ).hexdigest()


def _extract_ancestor_chain(
    node_name: str,
    run_index: int,
    run_data: Dict[str, List[Any]],
    max_depth: int = 50,
) -> List[Tuple[str, int, int]]:
    """Walk backward through source chain to build ancestor list.

    Args:
        node_name: Starting node name
        run_index: Starting run index
        run_data: Full execution runData
        max_depth: Maximum ancestor depth (prevents infinite loops)

    Returns:
        List of (node_name, run_index, distance) tuples, ordered by distance
    """
    ancestors = []
    visited: Set[Tuple[str, int]] = set()
    queue = [(node_name, run_index, 0)]  # (node, run, distance)

    while queue and len(ancestors) < max_depth:
        current_node, current_run, distance = queue.pop(0)

        # Prevent cycles
        key = (current_node, current_run)
        if key in visited:
            continue
        visited.add(key)

        # Get node runs
        node_runs = run_data.get(current_node, [])
        if current_run >= len(node_runs):
            continue

        run = node_runs[current_run]

        # Add to ancestors (excluding self at distance 0)
        if distance > 0:
            ancestors.append((current_node, current_run, distance))

        # Walk to parents via source chain
        if hasattr(run, "source") and run.source:
            for source in run.source:
                if source.previousNode:
                    parent_node = source.previousNode
                    parent_run = (
                        source.previousNodeRun
                        if source.previousNodeRun is not None
                        else 0
                    )
                    queue.append((parent_node, parent_run, distance + 1))

    logger.debug(
        f"Extracted {len(ancestors)} ancestors for {node_name}[{run_index}]"
    )
    return ancestors


def resolve_prompt_for_generation(
    node_name: str,
    run_index: int,
    run_data: Dict[str, List[Any]],
    prompt_registry: Dict[Tuple[str, int], PromptMetadata],
    agent_input: Optional[Any] = None,
) -> PromptResolutionResult:
    """Resolve prompt metadata for a generation span.

    Strategy:
    1. Walk ancestor chain backward from generation node
    2. Find all ancestors present in prompt_registry
    3. If exactly 1: use it (high confidence)
    4. If multiple: use fingerprint matching (medium confidence)
    5. If none: no prompt metadata

    Args:
        node_name: Generation node name
        run_index: Generation run index
        run_data: Full execution runData
        prompt_registry: Prompt fetch registry from prompt_detection module
        agent_input: Optional agent input for fingerprint matching

    Returns:
        PromptResolutionResult with metadata and debug info
    """
    # Extract ancestor chain
    ancestors = _extract_ancestor_chain(node_name, run_index, run_data)

    # Find prompt-fetch ancestors
    prompt_candidates: List[Tuple[PromptMetadata, int]] = []
    for ancestor_node, ancestor_run, distance in ancestors:
        key = (ancestor_node, ancestor_run)
        if key in prompt_registry:
            prompt_meta = prompt_registry[key]
            prompt_candidates.append((prompt_meta, distance))
            logger.debug(
                f"Found prompt candidate: {ancestor_node}[{ancestor_run}] "
                f"at distance {distance} "
                f"(name='{prompt_meta.name}', version={prompt_meta.version})"
            )

    # No candidates: no prompt metadata
    if not prompt_candidates:
        logger.debug(
            f"No prompt ancestors found for {node_name}[{run_index}]"
        )
        return PromptResolutionResult(
            resolution_method="none", candidate_count=0
        )

    # Single candidate: use it (high confidence)
    if len(prompt_candidates) == 1:
        prompt_meta, distance = prompt_candidates[0]
        logger.info(
            f"Resolved prompt for {node_name}[{run_index}]: "
            f"name='{prompt_meta.name}', version={prompt_meta.version}, "
            f"distance={distance}, confidence=high"
        )
        return PromptResolutionResult(
            prompt_name=prompt_meta.name,
            prompt_version=prompt_meta.version,
            original_version=prompt_meta.version,
            resolution_method="ancestor",
            confidence="high",
            ancestor_distance=distance,
            candidate_count=1,
            ambiguous=False,
        )

    # Multiple candidates: prefer closest OR use fingerprint matching
    prompt_candidates.sort(key=lambda x: x[1])  # Sort by distance
    closest_meta, closest_distance = prompt_candidates[0]

    # Check if closest is significantly closer than others
    second_distance = prompt_candidates[1][1]
    if closest_distance < second_distance:
        # Clear winner by distance
        logger.info(
            f"Resolved prompt for {node_name}[{run_index}]: "
            f"name='{closest_meta.name}', version={closest_meta.version}, "
            f"distance={closest_distance}, confidence=medium (closest of "
            f"{len(prompt_candidates)} candidates)"
        )
        return PromptResolutionResult(
            prompt_name=closest_meta.name,
            prompt_version=closest_meta.version,
            original_version=closest_meta.version,
            resolution_method="ancestor",
            confidence="medium",
            ancestor_distance=closest_distance,
            candidate_count=len(prompt_candidates),
            ambiguous=True,
        )

    # Tie in distance: try fingerprint matching
    if agent_input:
        best_match = _fingerprint_match(
            agent_input, [meta for meta, _ in prompt_candidates]
        )
        if best_match:
            logger.info(
                f"Resolved prompt for {node_name}[{run_index}] via "
                f"fingerprint matching: name='{best_match.name}', "
                f"version={best_match.version}, confidence=medium"
            )
            return PromptResolutionResult(
                prompt_name=best_match.name,
                prompt_version=best_match.version,
                original_version=best_match.version,
                resolution_method="fingerprint",
                confidence="medium",
                ancestor_distance=closest_distance,
                candidate_count=len(prompt_candidates),
                ambiguous=True,
            )

    # Fallback: use closest
    logger.warning(
        f"Ambiguous prompt resolution for {node_name}[{run_index}]: "
        f"{len(prompt_candidates)} equidistant candidates. "
        f"Using closest: '{closest_meta.name}' v{closest_meta.version}"
    )
    return PromptResolutionResult(
        prompt_name=closest_meta.name,
        prompt_version=closest_meta.version,
        original_version=closest_meta.version,
        resolution_method="ancestor",
        confidence="low",
        ancestor_distance=closest_distance,
        candidate_count=len(prompt_candidates),
        ambiguous=True,
    )


def _extract_prompt_text_from_input(agent_input: Any) -> Optional[str]:
    """Extract prompt/system message text from agent input.

    Handles various n8n agent input structures to find the system message
    or prompt text for fingerprint comparison.

    Args:
        agent_input: Agent input data (dict, list, or nested structure)

    Returns:
        Extracted text or None
    """
    if not agent_input:
        return None

    # If string: return as-is
    if isinstance(agent_input, str):
        return agent_input

    # If dict: search for common keys
    if isinstance(agent_input, dict):
        # Common keys for system messages
        for key in [
            "systemMessage",
            "system_message",
            "system",
            "prompt",
            "messages",
        ]:
            if key in agent_input:
                value = agent_input[key]
                if isinstance(value, str):
                    return value
                # Recurse for nested structures
                if isinstance(value, (dict, list)):
                    nested = _extract_prompt_text_from_input(value)
                    if nested:
                        return nested

    # If list: try first item (common in messages arrays)
    if isinstance(agent_input, list) and agent_input:
        for item in agent_input:
            text = _extract_prompt_text_from_input(item)
            if text:
                return text

    return None


def _fingerprint_match(
    agent_input: Any, candidates: List[PromptMetadata]
) -> Optional[PromptMetadata]:
    """Match agent input against prompt candidates using text fingerprinting.

    Compares fingerprints of agent input text against prompt text stored
    in candidate metadata. Returns best match if similarity exceeds threshold.

    Args:
        agent_input: Agent input data
        candidates: List of candidate PromptMetadata objects

    Returns:
        Best matching PromptMetadata or None
    """
    # Extract text from agent input
    input_text = _extract_prompt_text_from_input(agent_input)
    if not input_text or len(input_text) < 50:
        logger.debug("Agent input too short or missing for fingerprint matching")
        return None

    input_fp = _compute_text_fingerprint(input_text)

    # Compare against candidates
    # NOTE: This requires prompt text to be stored in PromptMetadata
    # Currently we don't store full prompt text (only metadata)
    # For now, this is a placeholder - would need to fetch prompt text
    # from registry or Langfuse API
    logger.debug(
        "Fingerprint matching not fully implemented (prompt text "
        "not available in registry)"
    )
    return None
