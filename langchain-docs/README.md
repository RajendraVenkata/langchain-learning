# LangChain Documentation (offline markdown)

Source: <https://docs.langchain.com/oss/python/langchain> — fetched via the per-page
`.md` export ("View as Markdown") on 2026-06-10.

Folder structure mirrors the docs site navigation. Files are numbered per folder to
preserve sidebar/reading order. LangGraph docs will follow the same layout under
`langgraph/`.

```
langchain-docs/
└── python/
    └── langchain/
        ├── 01-overview … 34-changelog-py   (top-level pages, sidebar order)
        ├── middleware/    (index, overview, built-in, custom)
        ├── streaming/     (index, overview, frontend)
        ├── frontend/      (overview … headless-tools)
        │   └── integrations/ (overview, ai-elements, assistant-ui, copilotkit, openui)
        ├── multi-agent/   (index, router, handoffs, subagents, skills, custom-workflow, …)
        └── test/          (index, unit-testing, integration-testing, evals)
```

## Top-level pages

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

## Notes

- Section landing pages (`middleware/00-index`, `streaming/00-index`) redirect to their
  `overview` page on the site, so those two pairs are intentionally identical copies.
- `frontend/00-overview` and `streaming/02-frontend` are the same page (cross-linked in nav).
- Content is the raw Mintlify markdown, so it contains components like `<Tip>`, `<CodeGroup>`,
  and `<Callout>` — these render as plain text but preserve all the prose and code examples.

Total: 72 markdown files.
