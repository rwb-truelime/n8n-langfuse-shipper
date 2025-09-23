from langfuse import Langfuse
import os

# Use env vars (fall back to dummy values for local/dev); avoid hard-coding real secrets in tests.
langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY", "sk-test"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY", "pk-test"),
    host=os.getenv("LANGFUSE_HOST", "http://localhost:3000"),
)

# Simple working trace with one child span and one generation
with langfuse.start_as_current_span(
    name="connection-test",
    metadata={"purpose": "sdk-connectivity"}
):
    with langfuse.start_as_current_span(
        name="sample-span",
        input={"message": "hello"},
        output={"response": "world"}
    ):
        pass

    # Use new unified observation API (generation type) to avoid deprecated helper
    with langfuse.start_as_current_observation(
        name="sample-generation",
        as_type="generation",
        input="What is 2+2?",
        output="4",
        model="dummy-model",
        metadata={"observation_type": "generation", "model": "dummy-model"}
    ):
        pass

# Ensure buffered data is sent
langfuse.flush()