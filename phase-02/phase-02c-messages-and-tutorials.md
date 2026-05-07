# Phase 02C — LangChain Core: Messages, Chat Models & Tutorials

> **Level:** Intermediate  
> **Part:** 3 of 3 in Phase 02  
> **Source files:** `migrate-complete.md` · `oss-complete.md`  
> **Prerequisite:** Complete Phase 02A and 02B first.
>
> ⚠️ **Note on examples:** Every code example in this file is extracted directly from your source files with inline source citations.

---

## Table of Contents

1. [Message Types](#1-message-types)
2. [Standard Content Blocks (v1 Feature)](#2-standard-content-blocks-v1-feature)
3. [Creating Multimodal Messages](#3-creating-multimodal-messages)
4. [Content Block Serialization](#4-content-block-serialization)
5. [Chat Model Initialization](#5-chat-model-initialization)
6. [Embedding Initialization](#6-embedding-initialization)
7. [Breaking Changes in Messages and Models](#7-breaking-changes-in-messages-and-models)
8. [Tutorials to Complete](#8-tutorials-to-complete)
9. [Phase 02 Complete Self-Quiz](#9-phase-02-complete-self-quiz)
10. [Phase 02 Master Flashcard Deck](#10-phase-02-master-flashcard-deck)
11. [Phase 02 Readiness Checklist](#11-phase-02-readiness-checklist)

---

## 1. Message Types

The conversation history is a list of message objects. Each message has a role and content.

From `migrate-complete.md` (Standard content section), the main message types are:

- `HumanMessage` — user turn
- `AIMessage` — model response turn
- `ToolMessage` — tool result turn
- `SystemMessage` — system-level instruction

---

## 2. Standard Content Blocks (v1 Feature)

In v1, all messages have a `content_blocks` property that provides a **provider-agnostic, standardised view** of message content.

**Source:** `migrate-complete.md` (Standard content → Read standardized content section)

```python
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5-nano")
response = model.invoke("Explain AI")

for block in response.content_blocks:
    if block["type"] == "reasoning":
        print(block.get("reasoning"))
    elif block["type"] == "text":
        print(block.get("text"))
```

**Key distinction from v0:**

- `message.content` — unchanged from v0. Returns strings or provider-native structures.
- `message.content_blocks` — **new in v1**. Returns standardised block dicts, regardless of provider.

---

## 3. Creating Multimodal Messages

Create messages with multiple content types (text + images).

**Source:** `migrate-complete.md` (Standard content → Create multimodal messages section)

```python
from langchain.messages import HumanMessage

message = HumanMessage(content_blocks=[
    {"type": "text", "text": "Describe this image."},
    {"type": "image", "url": "https://example.com/image.jpg"},
])
res = model.invoke([message])
```

**Example block shapes from `migrate-complete.md`:**

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
```

---

## 4. Content Block Serialization

By default, `content_blocks` are **not serialized** into the `content` field. If you need them in `content`, opt in.

**Source:** `migrate-complete.md` (Standard content → Serialize standard content section)

### Option A: Environment variable

```bash
export LC_OUTPUT_VERSION=v1
```

### Option B: Per-model parameter

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "gpt-5-nano",
    output_version="v1",
)
```

### Anthropic-specific

If you need the old behaviour for Anthropic:

```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(model="claude-sonnet-4-6", output_version="v0")
```

---

## 5. Chat Model Initialization

### `init_chat_model` — unified initialization

**Source:** Implied from namespace table in `migrate-complete.md` (Simplified package section)

```python
from langchain.chat_models import init_chat_model

# Initialize by model string (provider inferred)
model = init_chat_model("gpt-5-nano")
model = init_chat_model("claude-sonnet-4-6")
```

### `BaseChatModel` return type in v1

**Source:** `migrate-complete.md` (Breaking changes → Updated return type for chat models section)

The return type signature for chat model invocation is now `AIMessage` instead of `BaseMessage`:

```python
# v1 return type
def bind_tools(
        ...
    ) -> Runnable[LanguageModelInput, AIMessage]:
```

---

## 6. Embedding Initialization

### `init_embeddings` — unified initialization

**Source:** Implied from namespace table in `migrate-complete.md` (Simplified package section)

```python
from langchain.embeddings import init_embeddings

# Initialize by model string
embedder = init_embeddings("text-embedding-3-small")
```

---

## 7. Breaking Changes in Messages and Models

All message/model breaking changes in v1:

**Source:** `migrate-complete.md` (Breaking changes section)

| Change | What changed | Impact |
|---|---|---|
| `AIMessage.text` | Method → Property | `response.text()` becomes `response.text` |
| `AIMessage.example` | Removed | Use `additional_kwargs` instead |
| Chat model return type | `BaseMessage` → `AIMessage` | Custom models must update return type |
| `bind_tools` return type | `Runnable[..., BaseMessage]` → `Runnable[..., AIMessage]` | Custom models must update signature |
| `AIMessageChunk` | New `chunk_position` attribute | Indicates final chunk with `"last"` |
| OpenAI responses format | Various → `content` field | Standard location for responses |
| Anthropic `max_tokens` | Default was `1024` | Now defaults to higher values per model |
| Python version | 3.9+ supported | Now requires 3.10+ only |

### `AIMessage.text` property change

**Source:** `migrate-complete.md` (Minor changes section)

```python
# v1 — use as property
text = response.text

# v0 — old method (still works with warning)
text = response.text()
```

### `AIMessage.example` removal

**Source:** `migrate-complete.md` (Minor changes section)

```python
# ❌ No longer supported
AIMessage(content="Hello", example=True)

# ✅ Use additional_kwargs
AIMessage(content="Hello", additional_kwargs={"is_example": True})
```

### Anthropic `max_tokens` default

**Source:** `migrate-complete.md` (Breaking changes → Default max_tokens in langchain-anthropic section)

```python
from langchain_anthropic import ChatAnthropic

# If you need the old default
model = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=1024)
```

---

## 8. Tutorials to Complete

These four tutorials from `oss-complete.md` (Learn section) should be completed in Phase 02, in order:

### Tutorial 1: Semantic Search over PDF
**What you build:** A search engine querying a PDF with natural language.
**Key concepts:** Document loading, embeddings, vector stores, retrieval.

### Tutorial 2: RAG Agent
**What you build:** A full agent wrapping the semantic search in `create_agent`.
**Key concepts:** Connecting retrieval to agents, tool integration.

### Tutorial 3: SQL Agent
**What you build:** An agent writing and executing SQL with human-in-the-loop review.
**Key concepts:** Real middleware usage via `HumanInTheLoopMiddleware`.

### Tutorial 4: Voice Agent
**What you build:** A multimodal agent you can speak to.
**Key concepts:** Audio content blocks, multimodal `HumanMessage` construction.

---

## 9. Phase 02 Complete Self-Quiz

These 15 questions cover everything in Phase 02A, 02B, and 02C. Score 13/15 or higher before proceeding to Phase 03.

1. What is the new import path for `create_agent`? What was it in v0?
2. Did the `system_prompt` parameter exist in v0? If not, what was it called?
3. Can you pass a `ToolNode` to the `tools` parameter in v1? If not, what should you pass?
4. Name one error you SHOULD catch in `@wrap_tool_call`. Name one you SHOULD NOT.
5. What is the difference between `context=` and `messages=` when invoking an agent?
6. What method on `AgentMiddleware` do you implement to choose a different model per call?
7. In v0, how did you pass static metadata to an agent? In v1?
8. What types are supported for `state_schema` in v1? What types are no longer supported?
9. What is `message.content_blocks`? How does it differ from `message.content`?
10. How do you create a multimodal message in v1 with text and an image?
11. What changed about `AIMessage.text` in v1?
12. What are the two structured output strategies?
13. What is the new streaming node name in v1?
14. What method returns `None` to continue normally, or `dict` to update state, or `{"jump_to": "end"}` to short-circuit?
15. In the SQL Agent tutorial, which middleware is used to require human approval?

---

## 10. Phase 02 Master Flashcard Deck

25 core flashcards for spaced repetition.

| # | Question | Answer |
|---|---|---|
| 1 | Import path for `create_agent`? | `from langchain.agents import create_agent` |
| 2 | Old name of `system_prompt` parameter? | `prompt` |
| 3 | Does v1 accept `ToolNode`? | No — pass a list: `tools=[tool_a, tool_b]` |
| 4 | What must every `@tool` function have? | Docstring and type-annotated parameters |
| 5 | What replaced hooks? | Middleware with `before_model` / `after_model` |
| 6 | How to pass context in v1? | `context=Context(...)` on `invoke` / `stream` |
| 7 | Streaming node name in v1? | `"model"` (was `"agent"` in v0) |
| 8 | State type in v1? | `TypedDict` only (via `AgentState`) |
| 9 | What replaced prompted output? | `ToolStrategy` and `ProviderStrategy` |
| 10 | What does `{"jump_to": "end"}` do? | Short-circuits the agent loop immediately |
| 11 | Order of `before_model` with 3 middlewares? | List order: middleware[0] → middleware[1] → middleware[2] |
| 12 | Order of `after_model` with 3 middlewares? | Reverse: middleware[2] → middleware[1] → middleware[0] |
| 13 | Built-in middleware for conversation length? | `SummarizationMiddleware` |
| 14 | Built-in middleware for human approval? | `HumanInTheLoopMiddleware` |
| 15 | Which method chooses models dynamically? | `wrap_model_call` on `AgentMiddleware` |
| 16 | What is `message.content_blocks`? | Standardised content list (provider-agnostic, v1 only) |
| 17 | How to create multimodal message? | `HumanMessage(content_blocks=[{"type":"text",...}, {"type":"image",...}])` |
| 18 | What happened to `AIMessage.text`? | Became a property (not method) in v1 |
| 19 | What happened to `AIMessage.example`? | Removed — use `additional_kwargs` |
| 20 | How to serialize content_blocks? | `output_version="v1"` on model or `LC_OUTPUT_VERSION=v1` env var |
| 21 | What is `init_chat_model`? | Unified initializer for any chat model |
| 22 | What is `init_embeddings`? | Unified initializer for embeddings (RAG) |
| 23 | What changed with Anthropic defaults? | `max_tokens` now per-model (was always 1024) |
| 24 | Where is structured output generated? | In main loop (not separate node) |
| 25 | Name the 4 Phase 02 tutorials in order | Semantic Search → RAG Agent → SQL Agent → Voice Agent |

---

## 11. Phase 02 Readiness Checklist

✅ Complete this before Phase 03.

### Understanding
- [ ] I can explain `create_agent` and all key parameters
- [ ] I can define a `@tool` correctly (docstring + type hints)
- [ ] I understand static vs dynamic prompts
- [ ] I understand `context=` vs `messages=`
- [ ] I know middleware execution order (before/after)
- [ ] I understand `content_blocks` vs `content`
- [ ] I can create multimodal messages

### Migration knowledge
- [ ] I know all v0→v1 changes in Phase 02A §13
- [ ] I know all message breaking changes in §7 above

### Tutorials
- [ ] Semantic Search over PDF — completed
- [ ] RAG Agent — completed
- [ ] SQL Agent (with HITL) — completed
- [ ] Voice Agent — completed

### Self-assessment
- [ ] Phase 02 Complete Self-Quiz (§9) — 13/15 or higher
- [ ] All Phase 02 files saved to your storage system

### Ready for Phase 03?
- [ ] All checklist items ticked

---

> **Next:** Phase 03 — LangGraph Fundamentals  
> **Previous in Phase 02:** [Phase 02B — Middleware](./phase-02b-middleware.md) · [Phase 02A — Agents & Tools](./phase-02a-agents-and-tools.md)  
> **Source files used:** `migrate-complete.md` (messages, content blocks, breaking changes), `oss-complete.md` (tutorials list)
