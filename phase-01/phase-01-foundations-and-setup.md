# Phase 01 — Foundations & Environment Setup

> **Level:** Beginner  
> **Source files:** `oss-complete.md` · `migrate-complete.md`  
> **Goal:** Understand what the LangChain ecosystem is, what each package does, how to install everything correctly, and how to read version numbers and stability signals before writing a single line of agent code.

---

## Table of Contents

1. [The Ecosystem — What Are All These Packages?](#1-the-ecosystem)
2. [Installation & Version Management](#2-installation--version-management)
3. [Semantic Versioning — Reading Version Numbers](#3-semantic-versioning)
4. [API Stability Signals — What Is Safe to Use?](#4-api-stability-signals)
5. [Release Policy — LangChain, LangGraph & Deep Agents](#5-release-policy)
6. [Long-Term Support (LTS) — Who Gets Security Patches?](#6-long-term-support-lts)
7. [The 5 Core Namespaces in `langchain` v1](#7-the-5-core-namespaces)
8. [What Moved to `langchain-classic`?](#8-what-moved-to-langchain-classic)
9. [Security Mindset for LLM Applications](#9-security-mindset)
10. [Phase 01 — Self-Quiz](#10-self-quiz)
11. [Phase 01 — Flashcards](#11-flashcards)
12. [Phase 01 — Setup Checklist](#12-setup-checklist)

---

## 1. The Ecosystem

The LangChain ecosystem is **not a single library** — it is a family of packages that work together. Before writing any code you need to know which package does what.

| Package | What it does | Stability |
|---|---|---|
| `langchain-core` | The foundational primitives: messages, tools, base classes, runnables | Stable (1.0 LTS) |
| `langchain` | The main user-facing package: `create_agent`, middleware, chat models, embeddings | Stable (1.0 LTS) |
| `langgraph` | Graph-based agent orchestration: `StateGraph`, nodes, edges, checkpointers | Stable (1.0 LTS) |
| `langchain-community` | Community-maintained third-party integrations | Less strict versioning (see §6) |
| `langchain-openai` | LangChain-maintained partner package for OpenAI | Stable post 1.0 |
| `langchain-anthropic` | LangChain-maintained partner package for Anthropic | Stable post 1.0 |
| `langchain-classic` | Legacy patterns moved out of `langchain` v1: `LLMChain`, old retrievers, hub | Legacy — install only if needed |
| `deepagents` | Advanced agents with context management and virtual filesystem | Pre-1.0 — APIs may change |
| `langsmith` (SDK) | Observability, tracing, and deployment for agents | Separate product |

> **Key insight from your notes (`oss-complete.md`):** The `langchain` package namespace was **significantly reduced** in v1 to focus on essential building blocks. Legacy code was moved to `langchain-classic` to keep the main package clean.

---

## 2. Installation & Version Management

### Install the core stack

```bash
# Install the main packages
pip install -U langchain-core langchain langgraph

# Install a provider partner package (pick one or both)
pip install langchain-openai
pip install langchain-anthropic

# Install Deep Agents (optional, pre-1.0)
pip install deepagents
```

### Install to a specific version

```bash
# Pin to an exact version
pip install langchain-core==1.0.0
pip install langgraph==1.0.0
pip install deepagents==0.1.0
```

### Upgrade existing installations

```bash
# Upgrade LangChain
pip install -U langchain-core langchain

# Upgrade LangGraph
pip install -U langgraph langchain-core

# Upgrade Deep Agents
pip install -U deepagents
```

### Check your installed version

```python
# Check langchain-core version
import langchain_core
print(langchain_core.__version__)

# Check langgraph version
import langgraph
print(langgraph.__version__)
```

### Install the legacy package (only if needed)

```bash
pip install langchain-classic
```

> **When do you need `langchain-classic`?** Only if your code uses: legacy chains (`LLMChain`, `ConversationChain`), old retrievers (`MultiQueryRetriever`, `langchain.retrievers`), the indexing API, the hub module, or `langchain-community` re-exports from the old namespace. For new projects, you do **not** need it.

---

## 3. Semantic Versioning

All LangChain and LangGraph packages use the format `MAJOR.MINOR.PATCH` as defined by Semantic Versioning.

```
1  .  2  .  3
^     ^     ^
|     |     └── PATCH: bug fixes, security updates, doc improvements
|     └──────── MINOR: new features, backward-compatible enhancements
└────────────── MAJOR: breaking API changes, removal of deprecated features
```

### What each number means in practice

| Version bump | What it means for your code |
|---|---|
| `1.0.0` → `1.0.1` (PATCH) | Safe to upgrade — bug fixes only, nothing breaks |
| `1.0.0` → `1.1.0` (MINOR) | Safe to upgrade — new features added, existing code still works |
| `1.0.0` → `2.0.0` (MAJOR) | **Read the migration guide first** — breaking changes possible |

### Examples from your notes

- `1.0.0` — First stable release with production-ready APIs
- `1.1.0` — New features added in a backward-compatible manner
- `1.0.1` — Backward-compatible bug fixes

### Pre-release suffixes

| Suffix | Example | What it means |
|---|---|---|
| `a` (alpha) | `1.0.0a1` | Early preview — significant changes expected, use with caution |
| `b` (beta) | `1.0.0b1` | Feature-complete — minor changes possible, mostly safe |
| `rc` (release candidate) | `1.0.0rc1` | Final testing before stable release, very close to final |

---

## 4. API Stability Signals

When you look at LangChain documentation or source code, the **stability of an API is communicated through explicit markers**. Knowing these signals tells you whether it is safe to use something in production.

### Stable APIs (no prefix)
All APIs **without any special prefix** are stable and ready for production use. Breaking changes only happen in major releases (e.g. `2.0.0`).

### Beta APIs (`beta`)
- Feature-complete and tested
- Safe for production use with the understanding they may change
- Subject to minor API adjustments based on user feedback

### Alpha / Experimental APIs (`alpha`, `experimental`)
- Under active development
- May change significantly or be removed entirely
- Use with caution in production environments

### Deprecated APIs (`deprecated`)
- Will be removed in a future **major** release
- A migration guide is always provided alongside deprecation
- Continue to work throughout the entire current major version (e.g., all of `1.x`)
- Security updates are still provided during the deprecation period

### Internal APIs (`_` prefix)
- Functions or methods starting with a single underscore (`_`) are **private/internal**
- Not part of the public API — may change without any notice
- **Exception:** Methods prefixed with `_` that have no implementation and are meant to be overridden by subclasses — these *are* part of the public API despite the prefix

> **Rule of thumb from your notes:** If the documentation explicitly says something is internal, treat it as subject to change. If it starts with `_`, do not use it directly unless you are intentionally subclassing.

---

## 5. Release Policy

Understanding release cadence tells you **how often to expect updates** and whether to upgrade immediately or wait.

### LangChain (`langchain`, `langchain-core`)

| Release type | Cadence | What's included |
|---|---|---|
| **Major** | Infrequent | Breaking API changes, removal of deprecated features, architectural improvements |
| **Minor** | Frequently (multiple times) | New features, performance improvements, new optional parameters — no breaking changes |
| **Patch** | Up to a few times per week | Bug fixes, security updates, doc improvements — no API changes |

Key rule: **Breaking changes to the public API only occur in major version releases** (e.g., `2.0.0`).

### LangGraph (`langgraph`)

| Release type | Cadence | What's included |
|---|---|---|
| **Major** | At least 6–12 months apart | Breaking changes, removal of deprecated APIs |
| **Minor** | Every 1–2 months | New features and improvements |
| **Patch** | As needed, often weekly | Bug fixes and security issues |

### Deep Agents (`deepagents`) — Pre-1.0 rules apply

- **Minor** releases (e.g., `0.1.0` → `0.2.0`) **may contain breaking changes** while the package matures
- **Patch** releases (e.g., `0.1.0` → `0.1.1`) contain only bug fixes — no breaking changes
- Features marked `experimental` or `alpha` are subject to more significant changes
- Deep Agents will adopt the same LTS policies as LangChain and LangGraph **after reaching version 1.0**

> **What triggers a 1.0 release for Deep Agents?** According to your notes: when core APIs have stabilized based on community feedback, the package has been battle-tested in production environments, and breaking changes are no longer expected for core functionality.

### Deprecation process (all packages)

When any feature is deprecated:
1. **Deprecation Notice** — clearly marked with migration guidance
2. **Grace Period** — deprecated feature remains functional for at least one minor version
3. **Removal** — only happens in a major version release
4. **Migration Support** — migration guides provided, automated scripts when feasible

---

## 6. Long-Term Support (LTS)

LTS status tells you **which versions receive security patches and bug fixes**, and for how long.

### Status definitions

| Status | What it means |
|---|---|
| **ACTIVE** | Current development — gets bug fixes, security patches, and new features |
| **MAINTENANCE** | Gets security patches and critical bug fixes only — no new features |

### Current LTS status (from your notes)

| Package | Version | Status | Support Until |
|---|---|---|---|
| LangChain | 1.0 | **ACTIVE** | Until 2.0 is released |
| LangChain | 0.3 | **MAINTENANCE** | December 2026 |
| LangGraph | 1.0 | **ACTIVE** | Until 2.0 is released |
| LangGraph | 0.4 | **MAINTENANCE** | December 2026 |
| langchain-community | 0.4 | Special policy | See below |
| Deep Agents | pre-1.0 | Pre-release | N/A |

### After 2.0 is released (future)
Both LangChain 1.0 and LangGraph 1.0 will enter **MAINTENANCE mode for at least 1 year** after their respective 2.0 releases. This gives production applications time to migrate.

### Special case: `langchain-community`
`langchain-community` **does not follow the same strict semantic versioning** as `langchain` and `langchain-core`. Due to the nature of community contributions and third-party integrations, it may have breaking changes on minor releases. It has been released as version `0.4` to signal this different stability policy.

> **Practical implication:** Pin your `langchain-community` version more aggressively than you would pin `langchain` or `langchain-core`. Always test upgrades in a non-production environment first.

### Partner packages
Partner packages maintained by LangChain (such as `langchain-openai` and `langchain-anthropic`) follow semantic versioning and are expected to be stable post 1.0. Other partner packages may follow different policies — check their individual documentation.

---

## 7. The 5 Core Namespaces

In LangChain v1, the `langchain` package was **streamlined to 5 core namespaces**. Everything you need for building agents lives in these modules.

| Module | What's available | Notes |
|---|---|---|
| `langchain.agents` | `create_agent`, `AgentState` | Core agent creation |
| `langchain.messages` | Message types, content blocks, `trim_messages` | Re-exported from `langchain-core` |
| `langchain.tools` | `@tool` decorator, `BaseTool`, injection helpers | Re-exported from `langchain-core` |
| `langchain.chat_models` | `init_chat_model`, `BaseChatModel` | Unified model initialization |
| `langchain.embeddings` | `init_embeddings`, `Embeddings` | Embedding models |

### What each namespace gives you

**`langchain.agents`**
```python
from langchain.agents import create_agent, AgentState
```
This is where you build agents. `create_agent` is the primary function — it replaced `create_react_agent` from `langgraph.prebuilt` in v1.

**`langchain.messages`**
```python
from langchain.messages import HumanMessage, AIMessage, ToolMessage, trim_messages
```
Message types for constructing and manipulating conversation history.

**`langchain.tools`**
```python
from langchain.tools import tool, BaseTool
```
The `@tool` decorator turns a Python function into a tool an agent can call. `BaseTool` is the base class for more complex tools.

**`langchain.chat_models`**
```python
from langchain.chat_models import init_chat_model
```
A unified entry point to initialize any chat model regardless of provider.

**`langchain.embeddings`**
```python
from langchain.embeddings import init_embeddings
```
A unified entry point for embedding models used in RAG pipelines and vector stores.

---

## 8. What Moved to `langchain-classic`?

If you are coming from LangChain v0.x, some things you may have used are **no longer in the main `langchain` package**. They were moved to `langchain-classic` to keep the main package focused.

### Moved to `langchain-classic`

| What | Old import | New import |
|---|---|---|
| Legacy chains | `from langchain.chains import LLMChain` | `from langchain_classic.chains import LLMChain` |
| Old retrievers | `from langchain.retrievers import ...` | `from langchain_classic.retrievers import ...` |
| Indexing API | `from langchain.indexes import ...` | `from langchain_classic.indexes import ...` |
| Hub module | `from langchain import hub` | `from langchain_classic import hub` |

```bash
# Install if you need legacy functionality
pip install langchain-classic
```

> **Important:** For any new project, do **not** reach for `langchain-classic`. The modern equivalents (agents with `create_agent`, RAG via retriever tools, etc.) are in the main `langchain` package and are the patterns your tutor will teach you throughout this course.

---

## 9. Security Mindset

Because LangChain applications connect LLMs to external resources (file systems, databases, APIs), you need to adopt a security mindset **before** writing your first agent.

### The three core principles (from your notes)

**1. Limit permissions (Principle of Least Privilege)**
Scope permissions specifically to what the application needs. Never grant broad access.
- Use read-only credentials wherever possible
- Disallow access to sensitive resources by default
- Consider sandboxing agents inside containers
- Specify proxy configurations to control external requests

**2. Anticipate potential misuse**
LLMs can make mistakes, just like humans. Always assume that any system access or credentials **may be used in any way allowed by the permissions they are assigned**.

| Scenario | Risk | Mitigation |
|---|---|---|
| Agent with file system access | May delete or read sensitive files | Restrict to a specific directory; allow only safe file types |
| Agent with write API access | May write malicious data or delete records | Use read-only API keys; limit to safe endpoints |
| Agent with database access | May drop tables or mutate the schema | Scope credentials to only necessary tables; use READ-ONLY credentials |

**3. Defense in depth**
No single security technique is perfect. Combine multiple approaches rather than relying on any one layer.
- Use both read-only permissions **and** sandboxing
- Fine-tune prompts to reduce unsafe behavior (but do not rely on this alone)
- Design chains and agents with security constraints built in

### Reporting security issues

| Target | How to report |
|---|---|
| LangChain OSS libraries | Security advisory on GitHub + email `security@langchain.dev` |
| LangSmith | Email `security@langchain.dev` |

**Out of scope for bug bounties:** `langchain-experimental`, example code, example applications.

---

## 10. Self-Quiz

Test yourself before moving to Phase 02. Write your answers, then check them against the sections above.

1. Name all 5 core namespaces in `langchain` v1 and one thing available from each.
2. What is the pip command to upgrade LangChain to the latest version?
3. What does `MAINTENANCE` status mean for a package version?
4. A version bump from `1.2.0` to `1.3.0` is a what kind of release? Is it safe to upgrade without reading a migration guide?
5. Why does `langchain-community` have a different versioning policy from `langchain-core`?
6. What is the `_` prefix convention in Python, and what is the **exception** to this rule in LangChain?
7. You are building an agent that has access to a customer database. What three security mitigations should you apply?
8. Where does `LLMChain` live in LangChain v1? How do you import it?
9. What three conditions must Deep Agents meet before it will reach 1.0 and adopt LTS policies?
10. What is the difference between an `alpha` release and a `release candidate`?

---

## 11. Flashcards

Study these before starting Phase 02. Cover the **Answer** column and try to recall each answer from the **Question** alone.

| # | Question | Answer |
|---|---|---|
| 1 | What does `MAJOR.MINOR.PATCH` stand for in semantic versioning? | Major = breaking changes, Minor = new features (safe), Patch = bug fixes (safe) |
| 2 | Which version bump can introduce breaking API changes? | MAJOR only (e.g., `1.0.0` → `2.0.0`) |
| 3 | What does an `alpha` suffix (`1.0.0a1`) mean? | Early preview — significant changes expected, use with caution |
| 4 | What is `ACTIVE` LTS status? | Package receives bug fixes, security patches, AND new features |
| 5 | What is `MAINTENANCE` LTS status? | Package receives security patches and critical bug fixes only — no new features |
| 6 | Until when is LangChain 0.3 supported? | December 2026 (MAINTENANCE mode) |
| 7 | What package contains `create_agent` in v1? | `langchain.agents` |
| 8 | What package contains `StateGraph` in v1? | `langgraph` |
| 9 | What is `langchain-classic` for? | Legacy patterns removed from `langchain` v1 (LLMChain, old retrievers, hub, etc.) |
| 10 | Why is `langchain-community` versioned differently? | Community contributions + third-party integrations make strict semver impractical |
| 11 | What does a `_` prefix on a method mean? | Internal/private API — may change without notice |
| 12 | What is the Principle of Least Privilege in agent security? | Grant only the permissions the application strictly needs — no broader access |
| 13 | Name the 5 core `langchain` v1 namespaces | `agents`, `messages`, `tools`, `chat_models`, `embeddings` |
| 14 | What pip flag upgrades to the latest version? | `-U` (e.g., `pip install -U langchain-core langchain`) |
| 15 | Is `deepagents` stable? | No — it is pre-1.0; minor releases may have breaking changes |
| 16 | What is "defense in depth" in security? | Combining multiple security layers rather than relying on any single approach |
| 17 | When will LangChain 1.0 enter MAINTENANCE mode? | After LangChain 2.0 is released (then MAINTENANCE for at least 1 year) |
| 18 | What function replaced `create_react_agent`? | `create_agent` from `langchain.agents` |
| 19 | Where should you report a LangSmith security vulnerability? | `security@langchain.dev` |
| 20 | What are the three pre-release suffixes? | `a` (alpha), `b` (beta), `rc` (release candidate) |

---

## 12. Setup Checklist

Work through this list top to bottom. Check off each item as you complete it.

### Environment
- [ ] Python 3.10 or later is installed (`python --version`)
- [ ] A virtual environment is created and activated (`python -m venv .venv && source .venv/bin/activate`)

### Package installation
- [ ] `pip install -U langchain-core langchain langgraph` ran successfully
- [ ] `pip install langchain-openai` or `pip install langchain-anthropic` installed
- [ ] Version checks pass:
  ```python
  import langchain_core; print(langchain_core.__version__)
  import langgraph; print(langgraph.__version__)
  ```

### API keys
- [ ] OpenAI or Anthropic API key obtained and set as environment variable (`OPENAI_API_KEY` or `ANTHROPIC_API_KEY`)
- [ ] LangSmith API key obtained from [smith.langchain.com](https://smith.langchain.com) (you will need this in Phase 04)

### Knowledge verification
- [ ] You can name all 5 core `langchain` v1 namespaces from memory
- [ ] You can explain the difference between MAJOR, MINOR, and PATCH without looking
- [ ] You can explain what `ACTIVE` vs `MAINTENANCE` LTS status means
- [ ] You understand why `langchain-community` has looser versioning
- [ ] You have completed the Phase 01 Self-Quiz (§10) and reviewed any wrong answers

### Ready for Phase 02?
- [ ] All checklist items above are ticked
- [ ] You have saved this file to your chosen storage system (GitHub / Obsidian / local)

---

> **Next:** [Phase 02 — LangChain Core: Agents, Tools & RAG](./phase-02-langchain-core.md)  
> **Source notes used:** `oss-complete.md` (versioning, release policy, security, learn), `migrate-complete.md` (namespaces, langchain-classic, create_agent)
