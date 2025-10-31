"""Generation detection, usage extraction, concise output & Gemini anomaly.

All logic copied from legacy mapper.py sections to preserve behavior.
Any change requires test + docs updates.
"""
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import logging
from collections import deque

from ..models.langfuse import LangfuseUsage
from ..models.n8n import NodeRun

logger = logging.getLogger(__name__)

__all__ = [
    "GENERATION_PROVIDER_MARKERS",
    "detect_generation",
    "extract_usage",
    "extract_concise_output",
    "detect_gemini_empty_output_anomaly",
]

GENERATION_PROVIDER_MARKERS = [
    "openai", "anthropic", "gemini", "mistral", "groq",
    "lmchat", "lmopenai",
    "cohere", "deepseek", "ollama", "openrouter", "bedrock", "vertex", "huggingface", "xai",
    "limescape",
]


def _find_nested_key(data: Any, target: str, max_depth: int = 150) -> Optional[Dict[str, Any]]:
    try:
        if max_depth < 0:
            return None
        if isinstance(data, dict):
            if target in data:
                return data
            for v in data.values():
                found = _find_nested_key(v, target, max_depth - 1)
                if found is not None:
                    return found
        elif isinstance(data, list):
            for item in data[:150]:
                found = _find_nested_key(item, target, max_depth - 1)
                if found is not None:
                    return found
    except Exception:
        return None
    return None


def detect_generation(node_type: str, node_run: NodeRun) -> bool:
    if node_run.data and any(k in node_run.data for k in ("tokenUsage", "tokenUsageEstimate")):
        return True
    if any(
        _find_nested_key(node_run.data, k) is not None
        for k in ("tokenUsage", "tokenUsageEstimate")
    ):
        return True
    lowered = node_type.lower()
    if any(excl in lowered for excl in ("embedding", "embeddings", "reranker")):
        return False
    return any(marker in lowered for marker in GENERATION_PROVIDER_MARKERS)


def extract_usage(node_run: NodeRun) -> Optional[LangfuseUsage]:
    data = node_run.data or {}
    tu: Any = None
    for key in ("tokenUsage", "tokenUsageEstimate"):
        if isinstance(data, dict) and isinstance(data.get(key), dict):
            tu = data.get(key)
            break
    if tu is None:
        for key in ("tokenUsage", "tokenUsageEstimate"):
            container = _find_nested_key(data, key)
            if container and isinstance(container.get(key), dict):
                tu = container[key]
                break
    if not isinstance(tu, dict):
        def _scan_custom(obj: Any, depth: int = 25) -> Optional[Dict[str, Any]]:
            if depth < 0:
                return None
            if isinstance(obj, dict):
                keys = obj.keys()
                if any(k in keys for k in ("totalInputTokens", "totalOutputTokens", "totalTokens")):
                    return obj
                for v in list(obj.values())[:150]:
                    found = _scan_custom(v, depth - 1)
                    if found:
                        return found
            elif isinstance(obj, list):
                for item in obj[:150]:
                    found = _scan_custom(item, depth - 1)
                    if found:
                        return found
            return None
        tu = _scan_custom(data)
        if not isinstance(tu, dict):
            return None
    input_val = tu.get("input")
    output_val = tu.get("output")
    total_val = tu.get("total")
    p_tokens = tu.get("promptTokens")
    c_tokens = tu.get("completionTokens")
    t_tokens = (
        tu.get("totalTokens")
        or tu.get("totalInputTokens")
        or tu.get("totalOutputTokens")
    )
    if input_val is None:
        input_val = tu.get("totalInputTokens", input_val)
    if output_val is None:
        output_val = tu.get("totalOutputTokens", output_val)
    p_short = tu.get("prompt")
    c_short = tu.get("completion")
    if input_val is None:
        if p_tokens is not None:
            input_val = p_tokens
        elif p_short is not None:
            input_val = p_short
    if output_val is None:
        if c_tokens is not None:
            output_val = c_tokens
        elif c_short is not None:
            output_val = c_short
    if total_val is None:
        if t_tokens is not None:
            total_val = t_tokens
        elif input_val is not None and output_val is not None:
            try:
                total_val = int(input_val) + int(output_val)
            except Exception:
                pass
    if input_val is None and output_val is None and total_val is None:
        return None
    return LangfuseUsage(input=input_val, output=output_val, total=total_val)


def extract_concise_output(candidate: Any, node_type: str) -> Optional[str]:
    try:
        extracted_text: Optional[str] = None
        if isinstance(candidate, dict):
            resp = candidate.get("response") if isinstance(candidate.get("response"), dict) else candidate
            gens = resp.get("generations") if isinstance(resp, dict) else None
            if isinstance(gens, list) and gens:
                first_layer = gens[0]
                if isinstance(first_layer, list) and first_layer:
                    inner = first_layer[0]
                    if isinstance(inner, dict):
                        txt = inner.get("text")
                        if isinstance(txt, str) and txt.strip():
                            extracted_text = txt
        if (
            extracted_text is None
            and isinstance(candidate, dict)
            and "limescape" in node_type.lower()
        ):
            md_val = candidate.get("markdown")
            if isinstance(md_val, str) and md_val.strip():
                extracted_text = md_val
        if extracted_text is None and isinstance(candidate, dict):
            for k, v in candidate.items():
                if k.startswith("ai_") and isinstance(v, list) and v:
                    for layer in v[:3]:
                        if isinstance(layer, list) and layer:
                            first = layer[0]
                            if isinstance(first, dict):
                                txt = first.get("text")
                                if isinstance(txt, str) and txt.strip():
                                    extracted_text = txt
                                    break
                    if extracted_text:
                        break
        return extracted_text
    except Exception:
        return None


def detect_gemini_empty_output_anomaly(
    *,
    is_generation: bool,
    norm_output_obj: Any,
    run: NodeRun,
    node_name: str,
) -> Tuple[Optional[str], Dict[str, Any]]:
    if not is_generation:
        return None, {}
    meta: Dict[str, Any] = {}
    try:
        search_struct: Any = norm_output_obj if norm_output_obj is not None else run.data
        tu_source = search_struct if isinstance(search_struct, dict) else run.data
        tu = (
            (tu_source.get("tokenUsage") if isinstance(tu_source, dict) else None)
            or (tu_source.get("tokenUsageEstimate") if isinstance(tu_source, dict) else None)
            or {}
        )
        prompt_tokens = None
        completion_tokens = None
        total_tokens = None
        if isinstance(tu, dict):
            prompt_tokens = (
                tu.get("promptTokens")
                or tu.get("inputTokens")
                or tu.get("input_token_count")
            )
            completion_tokens = (
                tu.get("completionTokens")
                or tu.get("outputTokens")
                or tu.get("output_token_count")
            )
            total_tokens = tu.get("totalTokens") or tu.get("tokenCount")
        empty_text = False
        gen_block = search_struct
        if (
            isinstance(search_struct, dict)
            and "response" in search_struct
            and isinstance(search_struct["response"], dict)
        ):
            gen_block = search_struct["response"]
        try:
            gens = gen_block.get("generations") if isinstance(gen_block, dict) else None
            if isinstance(gens, list) and gens:
                first_layer = gens[0]
                if isinstance(first_layer, list) and first_layer:
                    inner = first_layer[0]
                    if isinstance(inner, dict):
                        txt = inner.get("text")
                        gen_info = inner.get("generationInfo")
                        if isinstance(txt, str) and txt == "":
                            empty_text = True
                        if isinstance(gen_info, dict) and len(gen_info) == 0:
                            meta["n8n.gen.empty_generation_info"] = True
        except Exception:
            pass
        anomaly = (
            empty_text
            and isinstance(prompt_tokens, int)
            and prompt_tokens > 0
            and (isinstance(total_tokens, int) and total_tokens >= prompt_tokens)
            and (completion_tokens in (0, None))
        )
        if anomaly:
            meta["n8n.gen.empty_output_bug"] = True
            meta["n8n.gen.prompt_tokens"] = prompt_tokens
            if total_tokens is not None:
                meta["n8n.gen.total_tokens"] = total_tokens
            meta["n8n.gen.completion_tokens"] = completion_tokens or 0
            if not run.error:
                run.error = {"message": "Gemini empty output anomaly detected"}
            try:
                logger.warning(
                    (
                        "gemini_empty_output_anomaly span=%s prompt_tokens=%s total_tokens=%s completion_tokens=%s"
                    ),
                    node_name,
                    prompt_tokens,
                    total_tokens,
                    completion_tokens,
                )
            except Exception:
                pass
            return "error", meta
    except Exception:
        return None, {}
    return None, meta
