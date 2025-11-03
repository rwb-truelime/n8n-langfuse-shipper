"""Prompt fetch node detection and metadata extraction.

This module identifies nodes that fetch prompts from Langfuse and extracts
their metadata (name, version, labels). Supports both HTTP Request nodes
calling Langfuse API and official @n8n/n8n-nodes-langchain.promptLangfuse.

Detection strategies:
1. Node type matching (@n8n/n8n-nodes-langchain.promptLangfuse)
2. HTTP Request to Langfuse prompt API endpoints
3. Output schema validation (name, version, prompt fields)
"""
from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel, Field, ValidationError

logger = logging.getLogger(__name__)


# Prompt fetch node type patterns
LANGFUSE_PROMPT_NODE_TYPES = [
    "@n8n/n8n-nodes-langchain.promptLangfuse",
]

# HTTP Request patterns for Langfuse API
LANGFUSE_PROMPT_API_PATTERNS = [
    "/api/public/prompts",
    "langfuse.com/api/public/prompts",
    "cloud.langfuse.com/api/public/prompts",
]


class PromptMetadata(BaseModel):
    """Metadata extracted from a Langfuse prompt fetch operation.

    Represents the core information needed to link a prompt to generation spans.
    """

    name: str
    version: int
    type: Optional[str] = None  # "text" or "chat"
    labels: List[str] = Field(default_factory=list)
    # Optional fingerprint for ambiguous case fallback (computed on demand)
    fingerprint: Optional[str] = None

    class Config:
        frozen = True  # Immutable for use as dict key


class PromptSourceInfo(BaseModel):
    """Complete information about a prompt fetch node and its output.

    Links node identity (name, run_index) to extracted prompt metadata.
    """

    node_name: str
    run_index: int
    node_type: str
    prompt_metadata: PromptMetadata
    detection_method: str  # "node_type" | "http_api" | "output_schema"


def _is_langfuse_prompt_node(node_type: str) -> bool:
    """Check if node type matches official Langfuse prompt node."""
    return node_type in LANGFUSE_PROMPT_NODE_TYPES


def _is_langfuse_http_request(
    node_type: str, parameters: Optional[Dict[str, Any]]
) -> bool:
    """Check if HTTP Request node calls Langfuse prompt API."""
    if node_type != "n8n-nodes-base.httpRequest":
        return False

    if not parameters:
        return False

    # Check URL parameter for Langfuse API endpoint
    url = parameters.get("url", "")
    if isinstance(url, str):
        return any(pattern in url for pattern in LANGFUSE_PROMPT_API_PATTERNS)

    return False


def _extract_prompt_metadata_from_output(
    output_data: Any,
) -> Optional[PromptMetadata]:
    """Extract prompt metadata from node output data.

    Handles multiple output formats:
    - Direct object: {"name": ..., "version": ..., "prompt": ...}
    - Wrapped in json key: {"json": {"name": ..., ...}}
    - Array of results: [{"json": {"name": ..., ...}}]
    - N8N data structure: {"main": [[{"json": {"name": ...}}]]}

    Returns None if output doesn't match expected schema.
    """
    if not output_data:
        return None

    # Unwrap common n8n output structures
    candidates = []

    # Direct object
    if isinstance(output_data, dict):
        candidates.append(output_data)
        # Check json wrapper
        if "json" in output_data:
            candidates.append(output_data["json"])
        # Check n8n main wrapper
        if "main" in output_data:
            main_data = output_data["main"]
            # main is typically [[{...}]]
            if isinstance(main_data, list) and main_data:
                for channel in main_data:
                    if isinstance(channel, list):
                        for item in channel:
                            if isinstance(item, dict):
                                candidates.append(item)
                                if "json" in item:
                                    candidates.append(item["json"])

    # Array of items (common in n8n outputs)
    if isinstance(output_data, list) and output_data:
        for item in output_data:
            if isinstance(item, dict):
                candidates.append(item)
                if "json" in item:
                    candidates.append(item["json"])

    # Try to validate each candidate
    for candidate in candidates:
        if not isinstance(candidate, dict):
            continue

        # Check for required fields
        if "name" not in candidate or "version" not in candidate:
            continue

        # Optional: validate presence of prompt text (not extracted here)
        # Could check for "prompt" or "config" keys

        try:
            return PromptMetadata(
                name=str(candidate["name"]),
                version=int(candidate["version"]),
                type=candidate.get("type"),
                labels=candidate.get("labels", []),
            )
        except (ValueError, TypeError, ValidationError) as e:
            logger.debug(
                f"Failed to parse prompt metadata from candidate: {e}"
            )
            continue

    return None


def detect_prompt_fetch_node(
    node_name: str,
    node_type: str,
    node_parameters: Optional[Dict[str, Any]],
    run_index: int,
    run_output: Any,
) -> Optional[PromptSourceInfo]:
    """Detect if node is a prompt fetch operation and extract metadata.

    Three-stage detection:
    1. Check node type (Langfuse prompt node)
    2. Check HTTP Request parameters (Langfuse API call)
    3. Validate output schema (prompt metadata present)

    Args:
        node_name: Name of the node in workflow
        node_type: Type identifier (e.g., @n8n/n8n-nodes-langchain.promptLangfuse)
        node_parameters: Node configuration parameters
        run_index: Execution run index (for multi-run nodes)
        run_output: Output data from node execution

    Returns:
        PromptSourceInfo if prompt detected, None otherwise
    """
    detection_method: Optional[str] = None

    # Stage 1: Node type detection
    if _is_langfuse_prompt_node(node_type):
        detection_method = "node_type"
        logger.debug(
            f"Node '{node_name}' detected as Langfuse prompt node by type"
        )

    # Stage 2: HTTP API detection
    elif _is_langfuse_http_request(node_type, node_parameters):
        detection_method = "http_api"
        logger.debug(
            f"Node '{node_name}' detected as Langfuse HTTP prompt fetch"
        )

    # If no type/API match, try output schema validation
    if not detection_method:
        # Only proceed if output looks promise-like
        prompt_metadata = _extract_prompt_metadata_from_output(run_output)
        if prompt_metadata:
            detection_method = "output_schema"
            logger.debug(
                f"Node '{node_name}' detected as prompt source by output "
                f"schema (name={prompt_metadata.name}, "
                f"version={prompt_metadata.version})"
            )
            return PromptSourceInfo(
                node_name=node_name,
                run_index=run_index,
                node_type=node_type,
                prompt_metadata=prompt_metadata,
                detection_method=detection_method,
            )
        return None

    # For node_type or http_api detections, extract metadata from output
    prompt_metadata = _extract_prompt_metadata_from_output(run_output)
    if not prompt_metadata:
        logger.warning(
            f"Node '{node_name}' matched prompt detection pattern "
            f"({detection_method}) but output does not contain valid "
            f"prompt metadata"
        )
        return None

    logger.info(
        f"Detected prompt fetch: node='{node_name}', run={run_index}, "
        f"name='{prompt_metadata.name}', version={prompt_metadata.version}, "
        f"method={detection_method}"
    )

    return PromptSourceInfo(
        node_name=node_name,
        run_index=run_index,
        node_type=node_type,
        prompt_metadata=prompt_metadata,
        detection_method=detection_method,
    )


def build_prompt_registry(
    run_data: Dict[str, List[Any]],
    workflow_nodes: List[Any],
) -> Dict[Tuple[str, int], PromptMetadata]:
    """Scan execution data and build registry of prompt fetches.

    Args:
        run_data: Execution runData mapping node names to run lists
        workflow_nodes: Static workflow node definitions

    Returns:
        Dictionary mapping (node_name, run_index) to PromptMetadata
    """
    registry: Dict[Tuple[str, int], PromptMetadata] = {}

    # Build lookup for node types and parameters
    node_info_map = {}
    for node in workflow_nodes:
        # Handle both dict and object nodes
        if isinstance(node, dict):
            node_name = node.get("name", "")
            node_type = node.get("type", "")
            node_parameters = node.get("parameters")
        else:
            node_name = getattr(node, "name", "")
            node_type = getattr(node, "type", "")
            node_parameters = getattr(node, "parameters", None)

        if node_name:
            node_info_map[node_name] = {
                "type": node_type,
                "parameters": node_parameters,
            }

    # Scan all node runs
    for node_name, runs in run_data.items():
        node_info = node_info_map.get(node_name, {})
        node_type = node_info.get("type", "")
        node_parameters = node_info.get("parameters")

        for run_index, run in enumerate(runs):
            # Extract output data (handle both dict and object runs)
            if isinstance(run, dict):
                run_output = run.get("data")
            else:
                run_output = getattr(run, "data", None)

            # Attempt detection
            source_info = detect_prompt_fetch_node(
                node_name=node_name,
                node_type=node_type,
                node_parameters=node_parameters,
                run_index=run_index,
                run_output=run_output,
            )

            if source_info:
                key = (node_name, run_index)
                registry[key] = source_info.prompt_metadata
                logger.debug(f"Registered prompt source: {key}")

    logger.info(
        f"Built prompt registry with {len(registry)} prompt fetch operations"
    )
    return registry
