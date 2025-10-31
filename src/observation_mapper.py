"""Map n8n workflow node types to Langfuse observation types.

This module provides a classification mechanism to map an n8n node's type and
category to a standardized Langfuse observation type (e.g., "generation",
"tool", "agent"). It uses a multi-tiered fallback strategy to determine the
most appropriate classification:

1.  **Exact Match:** Checks if the `node_type` is in a predefined set for a
    given observation type.
2.  **Regex Match:** If no exact match is found, it tests the `node_type`
    against a series of regular expressions.
3.  **Category Fallback:** As a final resort, it uses the node's `category`
    (e.g., "AI/LangChain Nodes") to infer a general observation type.

This logic is a direct port of the JavaScript implementation provided in the
project's reference materials, with minor Pythonic adjustments.
"""
from __future__ import annotations

from typing import Optional, Dict, Set

OBS_TYPES = [
    "agent",
    "tool",
    "chain",
    "retriever",
    "generation",
    "embedding",
    "evaluator",
    "guardrail",
    "event",
    "span",
]

EXACT_SETS: Dict[str, Set[str]] = {
    "agent": {"Agent", "AgentTool"},
    "generation": {
        "LmChatOpenAi",
        "LmOpenAi",
        "OpenAi",
        "Anthropic",
        "GoogleGemini",
        "Groq",
        "Perplexity",
        "LmChatAnthropic",
        "LmChatGoogleGemini",
        "LmChatMistralCloud",
        "LmChatOpenRouter",
        "LmChatXAiGrok",
        "OpenAiAssistant",
        # Custom extension: treat Limescape Docs node as generation regardless of heuristics.
        "limescapeDocs",
    },
    "embedding": {
        "EmbeddingsAwsBedrock",
        "EmbeddingsAzureOpenAi",
        "EmbeddingsCohere",
        "EmbeddingsGoogleGemini",
        "EmbeddingsGoogleVertex",
        "EmbeddingsHuggingFaceInference",
        "EmbeddingsMistralCloud",
        "EmbeddingsOllama",
        "EmbeddingsOpenAi",
    },
    "retriever": {
        "RetrieverContextualCompression",
        "RetrieverMultiQuery",
        "RetrieverVectorStore",
        "RetrieverWorkflow",
        "MemoryChatRetriever",
        "VectorStoreInMemory",
        "VectorStoreInMemoryInsert",
        "VectorStoreInMemoryLoad",
        "VectorStoreMilvus",
        "VectorStoreMongoDBAtlas",
        "VectorStorePGVector",
        "VectorStorePinecone",
        "VectorStorePineconeInsert",
        "VectorStorePineconeLoad",
        "VectorStoreQdrant",
        "VectorStoreSupabase",
        "VectorStoreSupabaseInsert",
        "VectorStoreSupabaseLoad",
        "VectorStoreWeaviate",
        "VectorStoreZep",
        "VectorStoreZepInsert",
        "VectorStoreZepLoad",
    },
    "evaluator": {"SentimentAnalysis", "TextClassifier", "InformationExtractor", "RerankerCohere", "OutputParserAutofixing"},
    "guardrail": {"GooglePerspective", "AwsRekognition"},
    "chain": {
        "ChainLlm",
        "ChainRetrievalQa",
        "ChainSummarization",
        "ToolWorkflow",
        "ToolExecutor",
        "ModelSelector",
        "OutputParserStructured",
        "OutputParserItemList",
        "OutputParserAutofixing",
        "TextSplitterCharacterTextSplitter",
        "TextSplitterRecursiveCharacterTextSplitter",
        "TextSplitterTokenSplitter",
        "ToolThink",
    },
}

REGEX_RULES = [
    ("agent", r"agent"),
    ("embedding", r"embedding"),
    ("retriever", r"(retriev|vectorstore)"),
    ("generation", r"(lmchat|^lm[a-z]|chat|openai|anthropic|gemini|mistral|groq|cohere)"),
    ("tool", r"tool"),
    ("chain", r"(chain|textsplitter|parser|memory)"),
    ("evaluator", r"(rerank|classif|sentiment|extract)"),
    ("guardrail", r"(perspective|rekognition|moderation|guardrail)"),
]

INTERNAL_LOGIC = {
    "If",
    "Switch",
    "Set",
    "Move",
    "Rename",
    "Wait",
    "WaitUntil",
    "Function",
    "FunctionItem",
    "Code",
    "NoOp",
    "ExecuteWorkflow",
    "SubworkflowTo",
}


def _category_fallback(node_type: str, category: Optional[str]) -> Optional[str]:
    """Determine observation type based on the node's category as a last resort.

    This function provides a fallback mapping when exact and regex matches fail.
    It uses the broad category assigned to a node in the n8n UI (e.g.,
    "AI/LangChain Nodes", "Transform Nodes") to make a general classification.

    Args:
        node_type: The specific type of the node (used for disambiguation within
            "Core Nodes").
        category: The category of the node, as defined in n8n.

    Returns:
        A string representing the inferred observation type, or None if no
        mapping is found.
    """
    if not category:
        return None
    match category:
        case "Trigger Nodes":
            return "event"
        case "Transform Nodes":
            return "chain"
        case "AI/LangChain Nodes":
            return "chain"
        case "Core Nodes":
            if node_type in INTERNAL_LOGIC:
                return "chain"
            if node_type in {"Schedule", "Cron"}:
                return "event"
            return "tool"
        case _:
            return None


def map_node_to_observation_type(node_type: Optional[str], category: Optional[str]) -> Optional[str]:
    """Map an n8n node to a Langfuse observation type using a fallback strategy.

    The function follows a specific precedence order to classify the node:
    1.  Checks for an exact match of `node_type` in `EXACT_SETS`.
    2.  If no exact match, searches for a regex match in `REGEX_RULES`.
    3.  If still no match, falls back to `_category_fallback` to infer from
        the node's `category`.

    Args:
        node_type: The type of the n8n node (e.g., "OpenAi", "If").
        category: The category of the node (e.g., "AI/LangChain Nodes").

    Returns:
        The corresponding Langfuse observation type as a string (e.g.,
        "generation", "chain"), or None if no classification could be made.
    """
    if not node_type:
        return None
    # 1. Exact set
    for obs_type, type_set in EXACT_SETS.items():
        if node_type in type_set:
            return obs_type
    # 2. Regex
    lower = node_type.lower()
    import re

    for obs_type, pattern in REGEX_RULES:
        if re.search(pattern, lower):
            return obs_type
    # 3. Category fallback
    return _category_fallback(node_type, category)


__all__ = ["map_node_to_observation_type", "OBS_TYPES"]

# ---------------------------------------------------------------------------
# AI Node Classification (Authoritative LangChain Package Mapping)
# ---------------------------------------------------------------------------
# All node *types* shipped inside the @n8n/n8n-nodes-langchain package are
# considered "AI" for the purposes of optional span filtering. The list below
# is derived from that package's package.json (nodes array). Keep the list
# synchronized if upstream adds or removes nodes. Sorting is alphabetical for
# deterministic diffs and easier visual scanning.
#
# NOTE: Some names (e.g. Code) also exist in Core categories; only the
# combination of type membership OR category "AI/LangChain Nodes" qualifies a
# node as AI. Category check provides a future‑proof fast path when present.
#
# Invariants (documented in copilot-instructions.md):
# * Adding / removing a type here requires test updates.
# * Helper function `is_ai_node` is pure and side‑effect free.
# * This module does NOT perform filtering; mapping layer consumes helper.
AI_NODE_TYPES: Set[str] = {
    # Agents & tools (agent ecosystem)
    "Agent",
    "AgentTool",
    "OpenAiAssistant",
    # Vendors / direct model wrappers
    "Anthropic",
    "GoogleGemini",
    "Ollama",
    "OpenAi",
    # Chain / higher-level orchestration
    "ChainLlm",
    "ChainRetrievalQa",
    "ChainSummarization",
    "SentimentAnalysis",
    "InformationExtractor",
    "TextClassifier",
    # Code execution scoped to AI pipelines
    "Code",
    # Document loaders
    "DocumentBinaryInputLoader",
    "DocumentDefaultDataLoader",
    "DocumentGithubLoader",
    "DocumentJsonInputLoader",
    # Embeddings
    "EmbeddingsAwsBedrock",
    "EmbeddingsAzureOpenAi",
    "EmbeddingsCohere",
    "EmbeddingsGoogleGemini",
    "EmbeddingsGoogleVertex",
    "EmbeddingsHuggingFaceInference",
    "EmbeddingsLemonade",
    "EmbeddingsMistralCloud",
    "EmbeddingsOllama",
    "EmbeddingsOpenAi",
    # LLM Chat variants
    "LmChatAnthropic",
    "LmChatAzureOpenAi",
    "LmChatAwsBedrock",
    "LmChatCohere",
    "LmChatDeepSeek",
    "LmChatGoogleGemini",
    "LmChatGoogleVertex",
    "LmChatGroq",
    "LmChatLemonade",
    "LmChatMistralCloud",
    "LmChatOllama",
    "LmChatOpenAi",
    "LmChatOpenRouter",
    "LmChatVercelAiGateway",
    "LmChatXAiGrok",
    # LLM Completion / other model wrappers
    "LmCohere",
    "LmLemonade",
    "LmOllama",
    "LmOpenAi",
    "LmOpenHuggingFaceInference",
    # MCP (Model Context Protocol tooling / triggers)
    "McpClientTool",
    "McpTrigger",
    # Memory nodes
    "MemoryBufferWindow",
    "MemoryChatRetriever",
    "MemoryManager",
    "MemoryMongoDbChat",
    "MemoryMotorhead",
    "MemoryPostgresChat",
    "MemoryRedisChat",
    "MemoryXata",
    "MemoryZep",
    # Output parsers
    "OutputParserAutofixing",
    "OutputParserItemList",
    "OutputParserStructured",
    # Rerankers / evaluators
    "RerankerCohere",
    # Retrievers & composite retrieval helpers
    "RetrieverContextualCompression",
    "RetrieverMultiQuery",
    "RetrieverVectorStore",
    "RetrieverWorkflow",
    # Text splitters
    "TextSplitterCharacterTextSplitter",
    "TextSplitterRecursiveCharacterTextSplitter",
    "TextSplitterTokenSplitter",
    # Tool nodes (invocable capabilities)
    "ToolCalculator",
    "ToolCode",
    "ToolHttpRequest",
    "ToolSearXng",
    "ToolSerpApi",
    "ToolThink",
    "ToolVectorStore",
    "ToolWikipedia",
    "ToolWolframAlpha",
    "ToolWorkflow",
    # Triggers / chat initiators
    "Chat",
    "ChatTrigger",
    "ManualChatTrigger",
    # Vector store management operations
    "VectorStoreInMemory",
    "VectorStoreInMemoryInsert",
    "VectorStoreInMemoryLoad",
    "VectorStoreMilvus",
    "VectorStoreMongoDBAtlas",
    "VectorStorePGVector",
    "VectorStorePinecone",
    "VectorStorePineconeInsert",
    "VectorStorePineconeLoad",
    "VectorStoreQdrant",
    "VectorStoreRedis",
    "VectorStoreSupabase",
    "VectorStoreSupabaseInsert",
    "VectorStoreSupabaseLoad",
    "VectorStoreWeaviate",
    "VectorStoreZep",
    "VectorStoreZepInsert",
    "VectorStoreZepLoad",
    # Utility orchestration
    "ToolExecutor",
    "ModelSelector",
    "Guardrails",
    # Custom extension already treated as generation
    "limescapeDocs",
}


_AI_NODE_TYPES_LOWER: Set[str] = {t.lower() for t in AI_NODE_TYPES}
_NAMESPACE_PREFIXES: tuple[str, ...] = (
    "@n8n/n8n-nodes-langchain.",
    "n8n-nodes-langchain.",
)


def _normalize_node_type(raw: str) -> str:
    """Normalize a raw node type for AI membership comparison.

    Steps:
    * Strip known namespace prefixes (JS package style).
    * Return lowercase for case-insensitive matching.
    * Preserve internal camel casing only for debugging purposes.
    """
    for p in _NAMESPACE_PREFIXES:
        if raw.startswith(p):  # exact prefix match
            raw = raw[len(p) :]
            break
    return raw.lower()


def is_ai_node(node_type: Optional[str], category: Optional[str]) -> bool:
    """Return True if the node is classified as an AI node.

    Detection precedence:
    1. Category equals "AI/LangChain Nodes" (fast path, future proof).
    2. Case-insensitive node type membership in `AI_NODE_TYPES`, allowing optional
       namespace prefixes (e.g. @n8n/n8n-nodes-langchain.OpenAi -> OpenAi).
    """
    if not node_type:
        return False
    if category == "AI/LangChain Nodes":
        return True
    return _normalize_node_type(node_type) in _AI_NODE_TYPES_LOWER


__all__.extend(["is_ai_node", "AI_NODE_TYPES"])
