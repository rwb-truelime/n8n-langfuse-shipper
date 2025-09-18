"""Map n8n workflow node types to Langfuse observation types.

Port of the provided JavaScript logic (langfuse-type-mapper.js) with minor Pythonic adjustments.
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
    ("chain", r"(chain|textsplitter|parser|memory|workflow)"),
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
