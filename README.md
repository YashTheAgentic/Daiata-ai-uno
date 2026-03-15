# Mini RAG System

A Retrieval-Augmented Generation pipeline built from scratch using individual libraries (no LangChain, no LlamaIndex). Processes 100 synthetic emails, retrieves relevant context via FAISS similarity search, and generates answers using OpenAI.

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Set your OpenAI API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY

# 3. Query via CLI (index is built automatically on first run)
python main.py "Who sent emails about budget approvals?"

# 4. Or use interactive mode
python main.py

# 5. Run evaluation
python eval.py
python eval.py --with-answers   # also shows generated answers
```

## Project Structure

```
mini-rag/
├── data/
│   └── generate_emails.py        # synthetic dataset generator
├── emails/                        # 100 synthetic email .txt files
├── rag/                           # core RAG components (built from scratch)
│   ├── chunker.py                   # document loading + text splitting
│   ├── embedder.py                  # OpenAI embedding API wrapper
│   ├── retriever.py                 # FAISS index build/load/search
│   └── generator.py                 # LLM prompt + answer generation
├── index/                         # pre-built FAISS index (gitignored)
├── main.py                        # CLI entry point
├── eval.py                        # retrieval evaluation (14 queries, precision@k)
├── requirements.txt               # dependencies
├── .env.example                   # API key template
├── .gitignore
└── README.md
```

## Architecture

```
User Query
    │
    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│  Embed Query │────▶│ FAISS Search │────▶│  Top-K Chunks │
│  (OpenAI)    │     │  (cosine)    │     │  + metadata   │
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
                                          ┌──────────────┐
                                          │    OpenAI     │
                                          │  + prompt     │
                                          └──────┬───────┘
                                                  │
                                                  ▼
                                            Answer with
                                           source citations
```

## Design Choices

### Chunking Strategy (`rag/chunker.py`)
- **Approach**: Character-based sliding window with sentence-boundary awareness.
- **Parameters**: `chunk_size=500`, `chunk_overlap=50`.
- **Rationale**: Emails average ~750 characters. At chunk size 500, most emails stay as a single chunk, preserving full context. The splitter looks for sentence boundaries (`.` `!` `?`) in the last 20% of each chunk to avoid cutting mid-sentence. Overlap of 50 chars ensures no information is lost at boundaries.
- **Metadata enrichment**: Each chunk is prefixed with the email's subject, sender, and receiver so the embedding captures the full context even for body-only fragments. This is a key design decision — without it, a body-only chunk like "I've prepared a detailed breakdown of all expenses" has no information about who sent it or what the email topic is.

### Embedding Model (`rag/embedder.py`)
- **Model**: OpenAI `text-embedding-3-small` (1536 dimensions).
- **Why**: High quality semantic understanding, fast inference, and cost-effective ($0.02/1M tokens). Vectors are L2-normalized at embedding time so inner product equals cosine similarity.
- **Batching**: Texts are embedded in batches of 100 to stay within rate limits while minimizing API round-trips.

### Vector Store (`rag/retriever.py`)
- **Library**: `faiss-cpu` with `IndexFlatIP` (inner product on normalized vectors = cosine similarity).
- **Why FAISS over alternatives**: Zero external dependencies, no database server needed, exact search is fast enough for 100 documents (~261 vectors). The index and chunk metadata are persisted to disk (`index/faiss.index` + `index/metadata.json`) so re-embedding is only needed when the corpus changes.
- **No approximate search**: With ~261 vectors, exact brute-force search is sub-millisecond. Approximate methods (IVF, HNSW) add complexity with no benefit at this scale.

### Generation (`rag/generator.py`)
- **Model**: Configurable via `OPENAI_MODEL_NAME` env var (defaults to `gpt-4o-mini`), `temperature=0` for deterministic, factual responses.
- **Prompt design**: The system prompt strictly grounds the LLM in the retrieved context, requires source citations (sender + subject), and instructs it to say "I don't have enough information" rather than hallucinate. This is critical for a RAG system where faithfulness to the source material matters.
- **No framework**: Uses the OpenAI Chat Completions API directly. The prompt is a simple system + user message pair — no chain abstraction needed.

### No End-to-End Frameworks
Every component is implemented directly:
- **Chunking**: Custom `_split_text()` function (~30 lines) with sentence-boundary logic.
- **Embedding**: Direct `openai.embeddings.create()` calls with numpy for normalization.
- **Retrieval**: Raw `faiss` index operations — `IndexFlatIP`, `add()`, `search()`.
- **Generation**: Direct `openai.chat.completions.create()` with a hand-written prompt template.

This gives full control over each stage and makes the tradeoffs explicit.

## Tradeoffs

| Decision | Chosen | Alternative | Why |
|---|---|---|---|
| Embedding model | OpenAI `text-embedding-3-small` | `all-MiniLM-L6-v2` (free, local) | Higher quality embeddings justify the small API cost; faster to implement |
| Chunk size | 500 chars | 200 tokens with overlap | Most emails fit in one chunk at 500; simpler, preserves full email context |
| Vector index | `IndexFlatIP` (exact) | `IndexIVFFlat` (approximate) | Exact search is sub-ms for ~261 vectors; approximate adds complexity for no gain |
| LLM | gpt-4o-mini | gpt-4o | Sufficient quality for email Q&A, ~10x cheaper, lower latency |
| Metadata storage | JSON file | SQLite / pickle | Human-readable, easy to inspect and debug; sufficient for 100 docs |
| Index persistence | Disk (faiss + json) | Rebuild every run | Saves embedding API costs and ~5s per startup |

## Evaluation Approach

### Methodology
The evaluation (`eval.py`) tests both **retrieval quality** and **generation faithfulness**. Retrieval is the most critical component: if retrieval fails, generation cannot succeed regardless of LLM quality. Generation is tested by printing the LLM's answer alongside retrieved sources so reviewers can visually verify faithfulness.

### Test Design
14 handcrafted queries covering all 9 email topics from the dataset specification:

| Topic | Queries |
|---|---|
| Budget Approval | 2 (who sent them, what they cover) |
| Training Opportunity | 1 |
| Technical Issue | 2 (general report, production incidents) |
| Meeting Request | 1 |
| Client Feedback | 2 (general feedback, praise vs. improvements) |
| Project Update | 2 (status updates, roadblocks) |
| Team Announcement | 1 |
| Deadline Extension | 2 (who requested, which projects) |
| Vendor Proposal | 1 |

### Metrics
- **Precision@k**: Fraction of top-k results that match expected subjects. Measures how many retrieved documents are relevant.
- **Hit@k**: Whether at least one correct result appears in top-k. Measures if the system finds *any* relevant document.
- Both computed at k=3 and k=5.

### Results
- **Avg Precision@3: 97.6%** | **Avg Precision@5: 97.1%** | **Hit Rate@3: 100%** | **Hit Rate@5: 100%**
- The one imperfect query ("Which projects need more time?") correctly retrieved Project Update emails alongside Deadline Extension emails — a reasonable semantic overlap, not a failure.

### Generation Verification
Running `python eval.py --with-answers` generates and displays the LLM's answer for each query alongside the retrieved sources, allowing manual inspection of faithfulness, citation accuracy, and appropriate scoping. All results are saved to `results.json`.

### Limitations
- Automated evaluation covers retrieval only. Generation faithfulness requires human review (enabled via `--with-answers`).
- The synthetic dataset has limited diversity — all emails on a topic share the same template body. This inflates retrieval scores compared to real-world data where topics overlap more.
- A more robust evaluation would use human-annotated relevance judgments and metrics like NDCG or MAP.

## Dependencies

| Package | Purpose |
|---|---|
| `openai` | Embeddings + LLM generation (direct API) |
| `faiss-cpu` | Vector similarity search |
| `numpy` | Vector normalization and array operations |
| `python-dotenv` | Environment variable management |

No end-to-end RAG frameworks are used. Each component is built from individual libraries.
