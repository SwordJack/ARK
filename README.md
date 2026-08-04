# Isomorphic Base (IsoBase)

Author: [SwordJack](https://github.com/SwordJack)

## README

- English [README.md](README.md)
- 汉语 [README.zh-cn.md](README.zh-cn.md)

## Opening

IsoBase started with a simple observation: most automation frameworks force you to
choose a vendor _before_ you write a single line of code. Your tools, your
storage, your LLM provider — every decision locks you deeper into a stack that
gets harder to change the further you go.

IsoBase takes the opposite approach. Build your logic first. Choose providers
later. Swap them when you need to.

It is a provider-neutral Python framework built around three pillars:

- A **JSON-driven workflow engine** — define execution trees declaratively; units persist state to SQLite so workflows can pause, resume, and recover across restarts.
- A **vendor-neutral LLM layer** — the same `ask()` / `generate()` / `generate_stream()` calls work identically on OpenAI Chat, Anthropic Messages, DashScope, or any compatible endpoint. Multi-turn tool calling, streaming, extended thinking, and web search come standard.
- A **pluggable knowledge base (RAG)** — index documents, embed chunks, retrieve via dense (cosine), sparse (BM25), or hybrid (RRF fusion) modes. Three storage backends (memory, SQL, Mongo). Zero extra dependencies for the core path.

Every module follows the same rule: _defaults stay simple, advanced features are opt-in_.

## Expectation

We believe automation should reduce complexity, not add it. IsoBase is designed
for the practitioner who needs a framework that adapts to their stack — not the
other way around.

- **Provider neutrality** means your code outlives your current vendor contract.
- **Storage portability** means you prototype on memory, ship on SQLite, and scale on PostgreSQL or MongoDB — without rewriting your retrieval logic.
- **Minimal dependencies** means you can install and run without pulling in a vector database, a Chinese tokenizer, or a cloud service you don't yet need.

These are design constraints we enforce in every pull request. If you want an
automation framework that respects your choices instead of making them for you,
you are welcome to use it — and help us keep improving it.

## Project Status

All six planned phases of the knowledge-base module are complete or accounted for:

| Phase                                                            | Status                          |
| ---------------------------------------------------------------- | ------------------------------- |
| Core interfaces, chunking, embedding, in-memory store            | ✅                              |
| SQL + Mongo stores, lifecycle APIs, shared contract tests        | ✅                              |
| Markdown structure-aware chunking (`heading_path`, front matter) | ✅                              |
| Retrieval enhancement (dense / sparse / hybrid / RRF / rerank)   | ✅                              |
| Document parsers                                                 | future                          |
| Vector index backends (pgvector / FAISS)                         | future performance optimization |

## Developers Guideline

We welcome automation enthusiasts around the globe to participate in the development of this open-source project. Please read the [Developer Guide](./docs/developer.md) to learn about the code conventions, unit testing, and version control protocol of this project.

## License

Copyright (c) 2024-2026 Landspark Digital Tech.

This project is licensed under the Mozilla Public License 2.0 - see the [LICENSE](LICENSE) file for details.
