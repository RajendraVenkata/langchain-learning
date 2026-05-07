# Phase 02C — LangChain Core: Messages, Chat Models & Tutorials

> **Level:** Intermediate  
> **Part:** 3 of 3 in Phase 02  
> **Source files:** `migrate-complete.md` · `oss-complete.md`  
> **Prerequisite:** Complete Phase 02A and 02B first.  
> **Goal:** Understand the message system (types, content blocks, multimodal), how to initialise chat models and embeddings, and which tutorials to complete in this phase.

---

## Table of Contents

1. [The Message System](#1-the-message-system)
2. [Standard Content Blocks (v1 Feature)](#2-standard-content-blocks)
3. [Multimodal Messages](#3-multimodal-messages)
4. [Serializing Content Blocks](#4-serializing-content-blocks)
5. [Chat Models — `init_chat_model`](#5-chat-models)
6. [Embeddings — `init_embeddings`](#6-embeddings)
7. [Breaking Changes in Messages (v1)](#7-breaking-changes)
8. [Tutorials to Complete in Phase 02](#8-tutorials)
9. [Phase 02 — Complete Self-Quiz (All 3 Files)](#9-complete-self-quiz)
10. [Phase 02 — Master Flashcard Deck](#10-master-flashcards)
11. [Phase 02 — Readiness Checklist](#11-readiness-checklist)

---

## 1. The Message System

Every agent conversation is a list of messages. LangChain defines specific message types for each role in the conversation.

### Message types

| Type | Import | When it appears | Key field |
|---|---|---|---|
| `HumanMessage` | `langchain.messages` | User turn | `content` |
| `AIMessage` | `langchain.messages` | Model response turn | `content`, `tool_calls` |
| `ToolMessage` | `langchain.messages` | Tool result turn | `content`, `tool_call_id` |
| `SystemMessage` | `langchain.messages` | System-level instruction | `content` |
| `AIMessageChunk` | `langchain.messages` | Streaming chunk from model | `content`, `chunk_position` |

```python
from langchain.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# Building a conversation manually
messages = [
    SystemMessage(content="You are a helpful assistant."),
    HumanMessage(content="What is the capital of France?"),
    AIMessage(content="The capital of France is Paris."),
    HumanMessage(content="And what about Germany?"),
]
```

### `trim_messages` — managing context window size

When conversation history grows long, use `trim_messages` to keep only what fits in the model's context window.

```python
from langchain.messages import trim_messages

trimmed = trim_messages(
    messages,
    max_tokens=1000,       # token budget
    strategy="last",       # keep the most recent messages
    token_counter=model,   # model used to count tokens
)
```

### Minor message changes in v1

**`AIMessage.text` is now a property (not a method):**
```python
# v1 — correct (property access)
text = response.text

# v0 — old (method call), still works but emits a deprecation warning
text = response.text()
```

**`AIMessageChunk` has a new `chunk_position` attribute:**
```python
for chunk in agent.stream({"messages": [...]}):
    if hasattr(chunk, "chunk_position") and chunk.chunk_position == "last":
        print("This is the final chunk in the stream")
```

**`AIMessage.example` parameter removed:**
```python
# ❌ No longer supported in v1
AIMessage(content="Hello", example=True)

# ✅ Use additional_kwargs for extra metadata
AIMessage(content="Hello", additional_kwargs={"is_example": True})
```

**Return type of chat model invocation fixed:**
```python
# v1 — return type is now explicitly AIMessage (not BaseMessage)
response: AIMessage = model.invoke("Hello")
```

---

## 2. Standard Content Blocks

This is a significant v1 feature. In v0, message content was **provider-specific** — OpenAI and Anthropic used different formats for the same content types (like reasoning/thinking output). In v1, messages have a `content_blocks` property that provides a **provider-agnostic, standardised view** of the content.

### The problem it solves

```python
# v0 — you had to write different code for each provider
response = model.invoke("Explain AI")
for item in response.content:
    if item.get("type") == "reasoning":
        ...  # OpenAI-style reasoning
    elif item.get("type") == "thinking":
        ...  # Anthropic-style thinking — different key!
    elif item.get("type") == "text":
        ...  # Text
```

### The v1 solution

```python
# v1 — provider-agnostic via content_blocks
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5-nano")
response = model.invoke("Explain AI")

# Works the same regardless of provider
for block in response.content_blocks:
    if block["type"] == "reasoning":
        print(block.get("reasoning"))
    elif block["type"] == "text":
        print(block.get("text"))
```

### Standard block shapes

```python
# Text block
text_block = {
    "type": "text",
    "text": "Hello world",
}

# Image block
image_block = {
    "type": "image",
    "url": "https://example.com/image.png",
    "mime_type": "image/png",
}

# Reasoning block (for models that support it)
reasoning_block = {
    "type": "reasoning",
    "reasoning": "Let me think through this step by step...",
}
```

### Key distinction

- `message.content` — unchanged from v0. Returns strings or provider-native structures. Safe for backward compatibility.
- `message.content_blocks` — **new in v1**. Returns a list of standardised block dicts, regardless of provider.

> **Rule of thumb:** Use `content_blocks` in new code when you need to process structured content (reasoning, images, text). Use `content` when you just need the raw text for display.

---

## 3. Multimodal Messages

Multimodal messages contain multiple content types in a single message (e.g., text + image). In v1, you construct these using `content_blocks` on `HumanMessage`.

```python
from langchain.messages import HumanMessage

# v1 — using content_blocks (provider-agnostic)
message = HumanMessage(content_blocks=[
    {"type": "text", "text": "What is in this image?"},
    {"type": "image", "url": "https://example.com/photo.jpg"},
])

response = model.invoke([message])
```

```python
# v0 — using content list (provider-specific format)
message = HumanMessage(content=[
    {"type": "text", "text": "What is in this image?"},
    {"type": "image_url", "image_url": {"url": "https://example.com/photo.jpg"}},  # OpenAI format
])
```

The v1 approach using `content_blocks` works across providers without changing the code.

---

## 4. Serializing Content Blocks

By default, standard content blocks in `content_blocks` are **not written back** into the `content` field. If you need them in `content` (e.g., when sending messages to a downstream client that reads `content`), opt in to serialization.

### Option A: Environment variable (affects all models in the process)

```bash
export LC_OUTPUT_VERSION=v1
```

### Option B: Per-model initialization parameter

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "gpt-5-nano",
    output_version="v1",    # serialize content_blocks into content
)
```

### Restoring v0 behaviour for OpenAI

If your code relies on the old OpenAI responses API format:

```python
from langchain_openai import ChatOpenAI

model = ChatOpenAI(model="gpt-5.4-mini", output_version="v0")
```

---

## 5. Chat Models

### `init_chat_model` — the unified initializer

`init_chat_model` is the recommended way to initialize any chat model in v1. It provides a single API regardless of the underlying provider.

```python
from langchain.chat_models import init_chat_model

# Initialize by model string (provider inferred)
model = init_chat_model("gpt-5-nano")
model = init_chat_model("claude-sonnet-4-6")
model = init_chat_model("gpt-5.4-mini")

# Basic invocation
response = model.invoke("What is the capital of India?")
print(response.text)   # property in v1, not method

# Streaming
for chunk in model.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)
```

### `BaseChatModel` — for custom implementations

If you are building a custom chat model (e.g., wrapping an internal API), subclass `BaseChatModel`. Its `bind_tools` return type has changed in v1:

```python
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage

class MyCustomModel(BaseChatModel):
    
    def bind_tools(self, tools, **kwargs) -> Runnable[LanguageModelInput, AIMessage]:
        # v1: return type is AIMessage, not BaseMessage
        ...
```

### `max_tokens` default change in `langchain-anthropic`

In v1, `langchain-anthropic` now defaults to higher `max_tokens` values based on the model. If your code relied on the old default of `1024`:

```python
from langchain_anthropic import ChatAnthropic

# Explicitly set max_tokens if you need the old behaviour
model = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1024)
```

---

## 6. Embeddings

### `init_embeddings` — the unified initializer

Used for creating embeddings for RAG pipelines and vector stores.

```python
from langchain.embeddings import init_embeddings

# Initialize an embedding model
embedder = init_embeddings("text-embedding-3-small")   # OpenAI
embedder = init_embeddings("embed-english-v3.0")        # Cohere

# Embed a single string
vector = embedder.embed_query("What is LangGraph?")

# Embed multiple documents
vectors = embedder.embed_documents([
    "LangGraph is a graph-based agent framework.",
    "LangChain provides tools for building LLM applications.",
])
```

### `Embeddings` base class

```python
from langchain.embeddings import Embeddings

class MyCustomEmbedder(Embeddings):
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        # your implementation
        ...
    
    def embed_query(self, text: str) -> list[float]:
        # your implementation
        ...
```

---

## 7. Breaking Changes in Messages (v1)

Summary of all message-related breaking changes, in one place for easy reference:

| Change | v0 behaviour | v1 behaviour |
|---|---|---|
| `AIMessage.text` | Method: `response.text()` | Property: `response.text` (method still works but warns) |
| `AIMessage.example` | Accepted as parameter | **Removed** — use `additional_kwargs` |
| Chat model return type | `BaseMessage` | `AIMessage` |
| `bind_tools` return type | `Runnable[..., BaseMessage]` | `Runnable[..., AIMessage]` |
| `AIMessageChunk` | No `chunk_position` | Has `chunk_position` attribute (`"last"` for final chunk) |
| OpenAI responses format | Stored in various places | Stored in message `content` by default |
| Anthropic `max_tokens` | Default `1024` | Default based on model (higher) |
| `content_blocks` | Not available | New property on all messages for provider-agnostic blocks |
| Multimodal messages | Provider-specific `content` list format | `content_blocks` list on `HumanMessage` |
| `LanguageModelOutputVar` | Typed as `BaseMessage` | Typed as `AIMessage` |
| AIMessageChunk merging | Simple ID selection | Prioritises provider-assigned IDs over LangChain-generated IDs |

---

## 8. Tutorials to Complete in Phase 02

These are the four LangChain tutorials listed in your notes (`oss-complete.md`). Complete them **in this order** — each one builds on the previous.

### Tutorial 1: Semantic Search over PDF

**What you build:** A semantic search engine that lets you query a PDF document using natural language.

**Key concepts covered:**
- Loading a PDF document
- Splitting text into chunks
- Creating embeddings with `init_embeddings`
- Storing embeddings in a vector store
- Querying the vector store with a natural language question
- Returning the most relevant chunks

**This is your introduction to the RAG pipeline components.** You are not yet building an agent — just the retrieval layer.

**Suggested approach:**
1. Ask your tutor: *"Walk me through the Semantic Search tutorial step by step, citing the source files."*
2. Build it yourself from scratch after the walkthrough.
3. Experiment: what happens if you change the chunk size? What if you embed the question differently?

---

### Tutorial 2: RAG Agent

**What you build:** A full RAG agent — the semantic search from Tutorial 1, but now wrapped in an agent using `create_agent`.

**Key concepts covered:**
- Turning the vector store retriever into a `@tool`
- Passing the retriever tool to `create_agent`
- The agent loop deciding when to call the retriever
- Combining retrieved context with the model's response

**This connects Phase 02A (agents/tools) with the RAG pipeline.** After this tutorial you will understand how retrieval fits inside an agent loop.

**Suggested approach:**
1. Ask your tutor: *"Show me how to convert the Tutorial 1 semantic search into a RAG agent using create_agent."*
2. Add a second tool (e.g., a web search fallback) and observe how the agent decides which tool to use.

---

### Tutorial 3: SQL Agent

**What you build:** An agent that writes and executes SQL queries against a database, with human-in-the-loop review before execution.

**Key concepts covered:**
- `@tool` for database query execution
- `HumanInTheLoopMiddleware` (from Phase 02B) applied to the SQL execution tool
- The agent composing SQL queries from natural language
- Reviewing and approving/rejecting queries before they run

**This is the first tutorial that uses the middleware system in a real project.** It solidifies your understanding of HITL patterns.

**Suggested approach:**
1. Ask your tutor: *"Walk me through the SQL Agent tutorial, focusing on how HumanInTheLoopMiddleware is applied."*
2. Try modifying the `interrupt_on` config to require approval only for `UPDATE` and `DELETE` queries, not `SELECT`.

---

### Tutorial 4: Voice Agent

**What you build:** An agent you can speak to and listen to — a multimodal conversational agent.

**Key concepts covered:**
- Multimodal `HumanMessage` construction with `content_blocks`
- Audio input and output content blocks
- How `content_blocks` enables provider-agnostic multimodal handling

**This puts Phase 02C's content blocks knowledge into practice.** After this tutorial you will have touched all four key areas of Phase 02.

**Suggested approach:**
1. Ask your tutor: *"Explain how the Voice Agent tutorial uses content_blocks for audio input/output."*
2. Focus on understanding the message structure before running the code.

---

## 9. Complete Self-Quiz (All 3 Files)

These 15 questions cover everything in Phase 02A, 02B, and 02C. If you can answer all 15 without looking, you are ready for Phase 03.

1. What function replaced `create_react_agent`? Where is it imported from?
2. Name three things `tools=` accepts and one thing it no longer accepts.
3. What is the difference between `system_prompt` and a dynamic prompt? How do you implement each?
4. What does `@wrap_tool_call` do? Give one example of a case where you should NOT catch the error.
5. What is the `context=` parameter for? How does it differ from `messages`?
6. What is the execution order of `before_model` and `after_model` when three middlewares are stacked?
7. What return value from `before_model` immediately ends the agent loop?
8. Describe two ways to define custom state in v1. When should you use each?
9. What is `message.content_blocks`? How does it differ from `message.content`?
10. How do you create a multimodal message in v1 that contains both text and an image?
11. What changed about `AIMessage.text` in v1?
12. What are the two structured output strategies that replaced prompted output?
13. What streaming node name should your code filter on in v1?
14. In the RAG Agent tutorial, how does the retriever become a tool the agent can use?
15. In the SQL Agent tutorial, which middleware is used to require human approval before executing queries?

---

## 10. Master Flashcard Deck

Combined flashcards for all of Phase 02. Study these using spaced repetition — after Phase 02, before Phase 03.

| # | Question | Answer |
|---|---|---|
| 1 | New import for `create_agent`? | `from langchain.agents import create_agent` |
| 2 | Old name of `system_prompt` parameter? | `prompt` |
| 3 | Does `tools=` accept `ToolNode`? | No — pass a plain list |
| 4 | What must every `@tool` function have? | A docstring and type-annotated parameters |
| 5 | What replaced pre/post-model hooks? | Middleware with `before_model` / `after_model` |
| 6 | How do you pass static context in v1? | `context=Context(...)` on `invoke` / `stream` |
| 7 | Streaming node name in v1? | `"model"` (was `"agent"` in v0) |
| 8 | What state type does `create_agent` support? | `TypedDict` only (via `AgentState`) |
| 9 | What replaced prompted structured output? | `ToolStrategy` and `ProviderStrategy` |
| 10 | What does `{"jump_to": "end"}` do in `before_model`? | Short-circuits the agent loop immediately |
| 11 | Order of `before_model` vs `after_model` in a stack? | `before_model`: list order. `after_model`: reverse order |
| 12 | Built-in middleware for conversation length? | `SummarizationMiddleware` |
| 13 | Built-in middleware for human approval? | `HumanInTheLoopMiddleware` |
| 14 | Which middleware method handles dynamic model selection? | `wrap_model_call` |
| 15 | Which middleware method wraps tool execution? | `wrap_tool_call` |
| 16 | What is `message.content_blocks`? | Provider-agnostic standardised content block list (new in v1) |
| 17 | How do you make a multimodal message in v1? | `HumanMessage(content_blocks=[{"type":"text",...}, {"type":"image",...}])` |
| 18 | What happened to `AIMessage.text`? | It became a property (not a method) in v1 |
| 19 | What happened to `AIMessage.example`? | Removed — use `additional_kwargs` |
| 20 | How do you serialize content_blocks into content? | `output_version="v1"` on the model, or `LC_OUTPUT_VERSION=v1` env var |
| 21 | What is `init_chat_model` for? | Unified initializer for any chat model provider |
| 22 | What is `init_embeddings` for? | Unified initializer for embedding models (used in RAG) |
| 23 | What changed with Anthropic `max_tokens` default? | Now defaults to higher values per model (was always 1024 in v0) |
| 24 | Where is structured output now generated in v1? | In the main loop (not a separate node) |
| 25 | Name the 4 Phase 02 tutorials in order | Semantic Search → RAG Agent → SQL Agent → Voice Agent |

---

## 11. Phase 02 Readiness Checklist

Work through this before starting Phase 03.

### Concepts
- [ ] I can explain `create_agent` and all its key parameters from memory
- [ ] I can define a `@tool` function correctly (docstring + type hints)
- [ ] I understand the difference between `system_prompt` and dynamic prompts
- [ ] I understand what `context=` is for and how it differs from `messages`
- [ ] I can explain the middleware execution order (before/after model)
- [ ] I know when to use `SummarizationMiddleware` vs `HumanInTheLoopMiddleware`
- [ ] I understand `content_blocks` and how it differs from `content`
- [ ] I can create a multimodal message using `content_blocks`

### Migration knowledge
- [ ] I know all the v0 → v1 changes in the quick reference table (Phase 02A, §10)
- [ ] I know all the message breaking changes (Phase 02C, §7)

### Tutorials
- [ ] Tutorial 1 completed: Semantic Search over PDF
- [ ] Tutorial 2 completed: RAG Agent
- [ ] Tutorial 3 completed: SQL Agent (with HITL)
- [ ] Tutorial 4 completed: Voice Agent

### Self-assessment
- [ ] Complete Phase 02 Self-Quiz (§9 in this file) — score 13/15 or higher before proceeding
- [ ] All Phase 02 files saved to your storage system (GitHub / Obsidian / local)

---

> **Next:** [Phase 03 — LangGraph Fundamentals](./phase-03-langgraph.md)  
> **Previous in Phase 02:** [Phase 02B — Middleware](./phase-02b-middleware.md) · [Phase 02A — Agents & Tools](./phase-02a-agents-and-tools.md)  
> **Source notes:** `migrate-complete.md` (messages, content blocks, breaking changes), `oss-complete.md` (tutorials list)
