# Phase 02C — LangChain Core: Messages, Chat Models & Tutorials

> **Level:** Intermediate  
> **Part:** 3 of 3 in Phase 02  
> **Source files:** `migrate-complete.md` · `oss-complete.md`  
> **Prerequisite:** Complete Phase 02A and 02B first.

---

## Table of Contents

1. [Message Types](#1-message-types)
2. [Standard Content Blocks](#2-standard-content-blocks)
3. [Creating Multimodal Messages](#3-creating-multimodal-messages)
4. [Content Block Serialization](#4-content-block-serialization)
5. [Chat Model Initialization](#5-chat-model-initialization)
6. [Embedding Initialization](#6-embedding-initialization)
7. [Breaking Changes in Messages](#7-breaking-changes-in-messages)
8. [Tutorials to Complete](#8-tutorials-to-complete)
9. [Phase 02 Complete Self-Quiz](#9-phase-02-complete-self-quiz)
10. [Phase 02 Master Flashcard Deck](#10-phase-02-master-flashcard-deck)
11. [Phase 02 Readiness Checklist](#11-phase-02-readiness-checklist)

---

## 1. Message Types

Every agent conversation is a list of message objects.

**Source:** `migrate-complete.md` (Standard content section)

### The Four Main Message Types

```python
from langchain.messages import HumanMessage, AIMessage, ToolMessage, SystemMessage

# User's turn
human_msg = HumanMessage(content="What's the weather?")

# Model's response
ai_msg = AIMessage(content="It's sunny")

# Tool's result (feedback from a tool call)
tool_msg = ToolMessage(
    content="Weather: Sunny, 25°C",
    tool_call_id="call_123"  # Links to the specific tool call
)

# System-level instruction
system_msg = SystemMessage(content="You are a weather assistant")

# A conversation is a list
messages = [system_msg, human_msg, ai_msg, tool_msg]
```

---

## 2. Standard Content Blocks

In v1, all messages have a `content_blocks` property that provides **provider-agnostic** content.

**Source:** `migrate-complete.md` (Standard content → Read standardized content section)

### Explanation

In v0, different providers returned different formats:
- OpenAI: `{"type": "reasoning", ...}`
- Anthropic: `{"type": "thinking", ...}`
- Same concept, different keys!

In v1, `content_blocks` standardises this:

```python
# Works the same for OpenAI, Anthropic, etc.
for block in response.content_blocks:
    if block["type"] == "text":
        print(block["text"])
    elif block["type"] == "reasoning":
        print("Reasoning:", block["reasoning"])
```

### Complete Runnable Code

```python
from langchain.chat_models import init_chat_model

# Initialize any chat model
model = init_chat_model("claude-sonnet-4-6")

# Get a response
response = model.invoke("Explain photosynthesis")

# Access standard content blocks (provider-agnostic)
print("Content blocks:")
for block in response.content_blocks:
    if block["type"] == "text":
        print(f"Text: {block['text']}")
    elif block["type"] == "reasoning":
        print(f"Reasoning: {block['reasoning']}")

# The same code works for OpenAI models too
model_openai = init_chat_model("gpt-5-mini")
response_openai = model_openai.invoke("Explain photosynthesis")

# Same loop works without changes!
for block in response_openai.content_blocks:
    if block["type"] == "text":
        print(f"Text: {block['text']}")
```

### content vs content_blocks

```python
response = model.invoke("Hello")

# content — old way (provider-specific)
print(response.content)  # Might be a string or dict, depends on provider

# content_blocks — new way (standardised)
for block in response.content_blocks:
    # Same structure across all providers
    if block["type"] == "text":
        print(block["text"])
```

---

## 3. Creating Multimodal Messages

Create messages with multiple content types (text + images).

**Source:** `migrate-complete.md` (Standard content → Create multimodal messages section)

### Explanation

Instead of having separate methods for text messages and image messages, use `content_blocks`:

```python
message = HumanMessage(content_blocks=[
    {"type": "text", "text": "Describe this:"},
    {"type": "image", "url": "https://..."}
])
```

### Complete Runnable Code

```python
from langchain.messages import HumanMessage
from langchain.chat_models import init_chat_model

# Create a multimodal message
message = HumanMessage(content_blocks=[
    {"type": "text", "text": "What do you see in this image?"},
    {"type": "image", "url": "https://upload.wikimedia.org/wikipedia/commons/4/4d/Cat_November_2010-1a.jpg"}
])

# Send to model
model = init_chat_model("gpt-5-mini")
response = model.invoke([message])

print(response.text)
# Output: "This image shows a brown cat..."

# Different image types
message_with_types = HumanMessage(content_blocks=[
    {"type": "text", "text": "Analyze these:"},
    {"type": "image", "url": "https://example.com/chart.png", "mime_type": "image/png"},
    {"type": "image", "url": "https://example.com/diagram.jpg", "mime_type": "image/jpeg"}
])

response = model.invoke([message_with_types])
print(response.text)
```

### Supported Block Types

```python
# Text block
{"type": "text", "text": "Hello"}

# Image block with URL
{"type": "image", "url": "https://example.com/image.png"}

# Image with explicit MIME type
{"type": "image", "url": "https://example.com/image.png", "mime_type": "image/png"}

# Reasoning block (if model supports it)
{"type": "reasoning", "reasoning": "Let me think through this..."}

# Create message with all types
message = HumanMessage(content_blocks=[
    {"type": "text", "text": "What's in this image?"},
    {"type": "image", "url": "https://example.com/pic.jpg"}
])
```

---

## 4. Content Block Serialization

By default, `content_blocks` are **not written back** into the `content` field. If downstream code needs them in `content`, opt in to serialization.

**Source:** `migrate-complete.md` (Standard content → Serialize standard content section)

### Explanation

```python
response = model.invoke("Hello")

# Without serialization (default)
response.content_blocks  # ← Standardised blocks available
response.content         # ← May not include what's in content_blocks

# With serialization
response.content_blocks  # ← Still available
response.content         # ← Now includes serialised content_blocks
```

### Enable Serialization

**Option 1: Environment variable** (affects all models in process)

```bash
export LC_OUTPUT_VERSION=v1
```

**Option 2: Per-model parameter**

```python
from langchain.chat_models import init_chat_model

model = init_chat_model(
    "gpt-5-mini",
    output_version="v1"  # ← Serialise content_blocks to content
)

response = model.invoke("Hello")
# Now response.content includes serialised blocks
```

**Option 3: Provider-specific (e.g., Anthropic)**

```python
from langchain_anthropic import ChatAnthropic

model = ChatAnthropic(
    model="claude-sonnet-4-6",
    output_version="v0"  # ← Use old format
)
```

### Complete Runnable Code

```python
from langchain.chat_models import init_chat_model

# Without serialization (default in v1)
model_default = init_chat_model("gpt-5-mini")
response = model_default.invoke("Explain AI")

print("Default behavior:")
print(f"content_blocks: {response.content_blocks}")  # ✅ Available
print(f"content: {response.content}")  # May not include all blocks

# With serialization enabled
model_serialized = init_chat_model(
    "gpt-5-mini",
    output_version="v1"
)
response = model_serialized.invoke("Explain AI")

print("\nWith serialization:")
print(f"content_blocks: {response.content_blocks}")  # ✅ Still available
print(f"content: {response.content}")  # ✅ Now includes serialised blocks
```

---

## 5. Chat Model Initialization

Use `init_chat_model` for a unified interface to any provider.

**Source:** `migrate-complete.md` (Simplified package section)

### Explanation

Instead of importing provider-specific classes, use a single function:

```python
# v0 — different import per provider
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI

# v1 — unified interface
from langchain.chat_models import init_chat_model

model = init_chat_model("gpt-5-mini")  # OpenAI
model = init_chat_model("claude-sonnet-4-6")  # Anthropic
model = init_chat_model("gemini-2.0-flash")  # Google
```

### Complete Runnable Code

```python
from langchain.chat_models import init_chat_model

# Initialize any model by string name (provider auto-detected)
model_openai = init_chat_model("gpt-5-mini")
model_anthropic = init_chat_model("claude-sonnet-4-6")
model_google = init_chat_model("gemini-2.0-flash")

# Basic invocation
response = model_openai.invoke("What is 2 + 2?")
print(response.text)  # Note: .text is a property, not .text()

# Streaming
for chunk in model_anthropic.stream("Tell me a story"):
    print(chunk.content, end="", flush=True)

# Configure parameters
model_custom = init_chat_model(
    "gpt-5-mini",
    temperature=0.7,      # More creative
    max_tokens=500,       # Limit response length
    output_version="v1"   # Serialise content_blocks
)

response = model_custom.invoke("Write a poem")
print(response.text)
```

---

## 6. Embedding Initialization

Use `init_embeddings` to initialize embedding models for RAG.

**Source:** `migrate-complete.md` (Simplified package section)

### Explanation

Embeddings convert text into vectors for similarity search (used in RAG pipelines).

```python
from langchain.embeddings import init_embeddings

embedder = init_embeddings("text-embedding-3-small")  # OpenAI
embedder = init_embeddings("embed-english-v3.0")      # Cohere
```

### Complete Runnable Code

```python
from langchain.embeddings import init_embeddings

# Initialize embedding model
embedder = init_embeddings("text-embedding-3-small")

# Embed a single query
query = "What is machine learning?"
query_vector = embedder.embed_query(query)
print(f"Query embedding size: {len(query_vector)}")  # e.g., 1536 dimensions

# Embed multiple documents (for a vector store)
documents = [
    "Machine learning is a subset of AI",
    "Deep learning uses neural networks",
    "Python is a popular language for ML"
]
doc_vectors = embedder.embed_documents(documents)
print(f"Got {len(doc_vectors)} embeddings")

# Now you'd store doc_vectors in a vector store like Pinecone, Weaviate, etc.
# And search by comparing query_vector to stored vectors
```

---

## 7. Breaking Changes in Messages

Summary of v0 → v1 message changes:

**Source:** `migrate-complete.md` (Breaking changes section)

| Change | v0 | v1 | Impact |
|---|---|---|---|
| `AIMessage.text` | Method: `response.text()` | Property: `response.text` | Update code |
| `AIMessage.example` | Supported parameter | Removed | Use `additional_kwargs` |
| Chat model return type | `BaseMessage` | `AIMessage` | Type hints more specific |
| `content_blocks` | Not available | New property | Access via `.content_blocks` |
| Anthropic `max_tokens` | Default: 1024 | Per-model default (higher) | May need explicit if you want 1024 |
| OpenAI responses format | Various locations | `content` field | More consistent |

### Breaking Change: AIMessage.text

```python
# v0 — called as method
response = model.invoke("Hello")
text = response.text()  # ← Called with ()

# v1 — property, not method
response = model.invoke("Hello")
text = response.text    # ← No () — it's a property
```

### Breaking Change: AIMessage.example

```python
# ❌ v0 — no longer supported
message = AIMessage(content="Hello", example=True)

# ✅ v1 — use additional_kwargs
message = AIMessage(
    content="Hello",
    additional_kwargs={"is_example": True}
)
```

### Breaking Change: Anthropic max_tokens Default

```python
from langchain_anthropic import ChatAnthropic

# v0 — always defaulted to 1024
model_v0 = ChatAnthropic(model="claude-sonnet-4-6")  # max_tokens=1024

# v1 — defaults to higher value (e.g., 4096)
model_v1 = ChatAnthropic(model="claude-sonnet-4-6")  # max_tokens=4096

# If you need old default
model_explicit = ChatAnthropic(
    model="claude-sonnet-4-6",
    max_tokens=1024  # Explicitly set
)
```

---

## 8. Tutorials to Complete

From `oss-complete.md` (Learn section), complete these 4 tutorials in order:

### Tutorial 1: Semantic Search over PDF ⭐

**What you build:** A search engine that queries a PDF with natural language.

**Concepts:**
- Load PDF documents
- Split into chunks
- Embed chunks with embeddings
- Store in vector database
- Search by similarity

**Why it matters:** Foundation for RAG systems. You'll use this in Tutorial 2.

### Tutorial 2: RAG Agent ⭐⭐

**What you build:** Wrap Tutorial 1 in an agent using `create_agent`.

**Concepts:**
- Turn retriever into a `@tool`
- Pass tool to agent
- Agent decides when to retrieve
- Combine retrieval with generation

**Why it matters:** RAG is the most common agent pattern. You're learning the core pattern.

### Tutorial 3: SQL Agent ⭐⭐⭐

**What you build:** Agent writes and executes SQL queries with human-in-the-loop approval.

**Concepts:**
- SQL execution as a tool
- `HumanInTheLoopMiddleware` for approval
- Agent composes natural language → SQL
- Human reviews before execution

**Why it matters:** First hands-on use of middleware from Phase 02B. Safety in agent actions.

### Tutorial 4: Voice Agent ⭐⭐⭐⭐

**What you build:** Agent you can speak to and listen to.

**Concepts:**
- Audio as content block
- Multimodal messages
- Speech-to-text and text-to-speech
- Provider-agnostic handling via `content_blocks`

**Why it matters:** Solidifies content blocks knowledge. Shows multimodal capabilities.

### Recommended Path

1. **Complete Tutorial 1** — Understand semantic search independently
2. **Complete Tutorial 2** — See how it integrates with agents
3. **Complete Tutorial 3** — Learn middleware safety patterns
4. **Complete Tutorial 4** — Touch multimodal capabilities

---

## 9. Phase 02 Complete Self-Quiz

Score 13/15 or higher before Phase 03.

1. What is the new import for `create_agent`? What was it in v0?
2. Name three types `tools=` accepts. Name one thing it no longer accepts.
3. Explain dynamic vs static prompts. When use each?
4. Name one error to catch in `@wrap_tool_call`. Name one NOT to catch.
5. What is `context=`? How does it differ from `messages=`?
6. What method on `AgentMiddleware` chooses models dynamically?
7. In v0, how did you pass metadata? In v1?
8. What state types work in v1? What no longer work?
9. What is `message.content_blocks`? How differs from `content`?
10. How do you create a multimodal message with text + image?
11. What changed about `AIMessage.text` in v1?
12. What two strategies replaced prompted output?
13. What streaming node should you filter on in v1?
14. What is `@dynamic_prompt`? When use it?
15. In SQL Agent tutorial, which middleware ensures approval?

---

## 10. Phase 02 Master Flashcard Deck

25 core flashcards. Study these before Phase 03.

| # | Question | Answer |
|---|---|---|
| 1 | Import for `create_agent` v1? | `from langchain.agents import create_agent` |
| 2 | Old name of `system_prompt`? | `prompt` |
| 3 | Three types `tools=` accepts? | `@tool` functions, plain callables, `BaseTool` instances |
| 4 | v0 pattern for static metadata? | `config={"configurable": {...}}` |
| 5 | v1 pattern for static metadata? | `context=Context(...)` |
| 6 | What replaced pre-model hooks? | Middleware with `before_model` method |
| 7 | What replaced post-model hooks? | Middleware with `after_model` method |
| 8 | State types in v1? | `TypedDict` only (via `AgentState`) |
| 9 | State types no longer supported? | Pydantic, dataclass |
| 10 | What is `content_blocks`? | Standardised content list (provider-agnostic) |
| 11 | How create multimodal message? | `HumanMessage(content_blocks=[{...}, {...}])` |
| 12 | What changed `AIMessage.text`? | Method → Property (no `()`) |
| 13 | What replaced prompted output? | `ToolStrategy` and `ProviderStrategy` |
| 14 | Streaming node in v1? | `"model"` (was `"agent"` in v0) |
| 15 | Two structured output strategies? | `ToolStrategy` (all models), `ProviderStrategy` (fast, model-specific) |
| 16 | Before/after_model order 3 MW? | before: list order (0→1→2), after: reverse (2→1→0) |
| 17 | Return value to short-circuit? | `{"jump_to": "end"}` from `before_model` |
| 18 | Middleware for conversation length? | `SummarizationMiddleware` |
| 19 | Middleware for approvals? | `HumanInTheLoopMiddleware` |
| 20 | Method for dynamic model selection? | `wrap_model_call` on `AgentMiddleware` |
| 21 | What is `init_chat_model`? | Unified initializer for any chat model |
| 22 | What is `init_embeddings`? | Unified initializer for embedding models |
| 23 | Phase 02 tutorials in order? | Semantic Search → RAG Agent → SQL Agent → Voice Agent |
| 24 | What is `@wrap_tool_call`? | Decorator for middleware that handles tool errors |
| 25 | What is `ToolRuntime`? | Parameter type for tools that access agent state |

---

## 11. Phase 02 Readiness Checklist

✅ Complete before Phase 03.

### Understanding ✓
- [ ] I can explain `create_agent` and all key parameters
- [ ] I can define a `@tool` correctly
- [ ] I understand static vs dynamic prompts
- [ ] I understand `context=` vs `messages=`
- [ ] I know middleware execution order
- [ ] I understand `content_blocks` vs `content`
- [ ] I can create multimodal messages

### Migration Knowledge ✓
- [ ] All v0 → v1 changes in §2 of Phase 02A
- [ ] All message breaking changes in §7 above

### Tutorials ✓
- [ ] Tutorial 1: Semantic Search — completed
- [ ] Tutorial 2: RAG Agent — completed
- [ ] Tutorial 3: SQL Agent — completed
- [ ] Tutorial 4: Voice Agent — completed

### Self-Assessment ✓
- [ ] Phase 02 Complete Self-Quiz (§9) — **13/15 or higher**
- [ ] All Phase 02 files downloaded/stored

### Ready for Phase 03? ✓
- [ ] All checkmarks above completed

---

> **Next:** Phase 03 — LangGraph Fundamentals  
> **Files in Phase 02:**
> - [Phase 02A — Agents & Tools](./phase-02a-agents-and-tools.md)
> - [Phase 02B — Middleware](./phase-02b-middleware.md)
> - [Phase 02C — Messages & Tutorials](./phase-02c-messages-and-tutorials.md)
