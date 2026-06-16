# LangChain & LangGraph Documentation (offline markdown)

Source: <https://docs.langchain.com/oss/python> — fetched via the per-page `.md` export
("View as Markdown"). LangChain fetched 2026-06-10, LangGraph fetched 2026-06-16.

Folder structure mirrors the docs site navigation. Files are numbered per folder to
preserve sidebar/reading order.

```
langchain-docs/
└── python/
    ├── langchain/      (72 files)
    │   ├── 01-overview … 34-changelog-py   (top-level pages, sidebar order)
    │   ├── middleware/  · streaming/  · multi-agent/  · test/
    │   └── frontend/      (+ integrations/ subfolder)
    ├── langgraph/      (40 files)
    │   ├── 01-overview … 37-changelog-py   (top-level pages, sidebar order)
    │   └── frontend/      (overview, graph-execution, custom-stream-channels)
    ├── deepagents/     (59 files)
    │   ├── 00-overview … 36-programmatic-subagents   (top-level pages)
    │   └── cli/  · code/  · deploy/  · frontend/  · streaming/
    ├── integrations/   (41 files)
    │   └── chat/ · document_loaders/ · embeddings/ · llms/ · providers/
    │         · retrievers/ · sandboxes/ · splitters/ · tools/ · vectorstores/ · …
    └── reference/      (landing page only — API docs live on reference.langchain.com)
        └── 00-overview.md  + README.md
```

---

## LangChain — top-level pages

| # | Page | # | Page |
|---|------|---|------|
| 01 | overview | 18 | retrieval |
| 02 | install | 19 | studio |
| 03 | quickstart | 20 | ui |
| 04 | philosophy | 21 | deploy |
| 05 | agents | 22 | observability |
| 06 | models | 23 | component-architecture |
| 07 | messages | 24 | harness |
| 08 | tools | 25 | knowledge-base |
| 09 | short-term-memory | 26 | rag |
| 10 | long-term-memory | 27 | sql-agent |
| 11 | event-streaming | 28 | supervisor |
| 12 | structured-output | 29 | voice-agent |
| 13 | runtime | 30 | deep-agent-from-scratch |
| 14 | context-engineering | 31 | evals |
| 15 | guardrails | 32 | academy |
| 16 | mcp | 33 | get-help |
| 17 | human-in-the-loop | 34 | changelog-py |

## LangGraph — top-level pages

| # | Page | # | Page |
|---|------|---|------|
| 01 | overview | 20 | studio |
| 02 | install | 21 | ui |
| 03 | quickstart | 22 | deploy |
| 04 | local-server | 23 | observability |
| 05 | thinking-in-langgraph | 24 | pregel |
| 06 | workflows-agents | 25 | choosing-apis |
| 07 | persistence | 26 | graph-api |
| 08 | checkpointers | 27 | use-graph-api |
| 09 | stores | 28 | functional-api |
| 10 | fault-tolerance | 29 | use-functional-api |
| 11 | event-streaming | 30 | durable-execution |
| 12 | streaming | 31 | timeout-and-error-handling |
| 13 | interrupts | 32 | human-in-the-loop |
| 14 | use-time-travel | 33 | memory |
| 15 | add-memory | 34 | agentic-rag |
| 16 | use-subgraphs | 35 | sql-agent |
| 17 | application-structure | 36 | case-studies |
| 18 | test | 37 | changelog-py |
| 19 | backward-compatibility | | |

## Notes

- Content is the raw Mintlify markdown, so it contains components like `<Tip>`, `<CodeGroup>`,
  and `<Callout>` — these render as plain text but preserve all the prose and code examples.
- **LangChain:** section landing pages (`middleware/00-index`, `streaming/00-index`) redirect to
  their `overview` page, so those pairs are intentionally identical; `frontend/00-overview` and
  `streaming/02-frontend` are the same cross-linked page.
- **Both `changelog-py` files** come from the shared `/oss/python/releases/changelog` page
  (the per-section `.md` URL returns HTML, so it was fetched from the canonical path).
- Some LangGraph pages share content where the docs cross-link them, e.g.
  `07-persistence` ≡ `30-durable-execution`, `10-fault-tolerance` ≡ `31-timeout-and-error-handling`,
  and `13-interrupts` ≡ `32-human-in-the-loop`.
- **Deep Agents:** the bare section landings `26-cli`/`27-code` redirect to `cli/00-overview` /
  `code/00-overview`, so those are duplicate copies (kept for completeness).
- **Integrations:** this mirrors only what the `/oss/python/integrations/` nav exposes (mostly
  Google/Azure/major providers). A handful of nav links (`chat_loaders/gmail`, `vectorstores/scann`)
  are redirect-only stubs that 404 as `.md` and were skipped. The full provider catalog lives on
  reference.langchain.com.
- **Reference:** only the landing page is here; the real API reference is a separate autodoc site
  (see `python/reference/README.md`).

Total: 213 markdown files — LangChain 72 + LangGraph 40 + Deep Agents 59 + Integrations 41 + Reference 1.
