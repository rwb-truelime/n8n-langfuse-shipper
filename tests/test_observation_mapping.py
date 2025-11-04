from __future__ import annotations

from n8n_langfuse_shipper.observation_mapper import map_node_to_observation_type


def test_observation_exact_sets():
    assert map_node_to_observation_type("OpenAi", None) == "generation"
    assert map_node_to_observation_type("EmbeddingsOpenAi", None) == "embedding"
    assert map_node_to_observation_type("VectorStoreQdrant", None) == "retriever"
    assert map_node_to_observation_type("Agent", None) == "agent"
    assert map_node_to_observation_type("ToolWorkflow", None) == "chain"


def test_observation_regex_fallbacks():
    # Should hit regex rules when not in exact sets
    assert map_node_to_observation_type("CustomLmChatAdapter", None) == "generation"
    assert map_node_to_observation_type("MyVectorStoreAdapter", None) == "retriever"
    assert map_node_to_observation_type("ToolAlpha", None) == "tool"
    assert map_node_to_observation_type("ChainifySomething", None) == "chain"


def test_observation_category_fallback():
    # Not matching exact or regex deliberately
    assert map_node_to_observation_type("If", "Core Nodes") == "chain"
    assert map_node_to_observation_type("Schedule", "Core Nodes") == "event"
    assert map_node_to_observation_type("UnknownType", "Trigger Nodes") == "event"
