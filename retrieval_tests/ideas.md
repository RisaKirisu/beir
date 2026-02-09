## RAG

Current: Contextual Retrieval: for each chunk, generate and concatenate a contextual description that summarizes its relevance in the parent document.

### Practical Considerations

- Multi-tenant - chats must be isolated. Most chats has no or little documents. Only very few chats have many/long documents.
- Indexing vector field is a must on large vector set, but HNSW doesn't work when pre-filtering is needed (i.e. on chat_id).
- Balance between accuracy, cost, latency. Modern LLMs are getting better at filtering noise.

### Ideas:

- Split documents into larger "generative" chunks (~1000 tokens) and smaller "retrieval" chunks (~200-300 tokens).
- For each "retrieval" chunk, generate an contextual global anchor that describes how the chunk situate in the full parent document and the unique contribution of the chunk.
- For each "retrieval" chunk, generate a set of questions that may be answered by the chunk.
- For each "retrieval" chunk, generate a list of keywords and key phrases that helps full text search to find the chunk.

### Implementation Idea:

- Separate tables: one table for "generative" chunks that store the full chunk text and id, an FTS table for "retrieval" chunks that store tsvectors for FTS index with GIN, a vector search table that stores vectors for similarity search indexed with hnsw if the size grow past a threshold. Rows of the latter 2 tables reference to the parent "generative" chunk id.
- The FTS table's tsvector column is caculated by weighted combination of keywords & key phrases, "retrieval" chunk text, generated questions, and contextual anchor, in this order.
- The vector table's vector column is calculated by embedding concatenated [contextual anchor + "retrieval" chunk text] and each of the generated questions. Generated perspectives are not embeded as embeddings already capture multi-dimensional sementics, while keywords are more helpful for text-based searches. Hence, each "retrival" chunk will have multiple vector entries in the vector table, and all of these entries references to their parent "retrieval" chunk id.
- Therefore, if a "generative" chunk has k "retrieval" chunks, there will be k rows in the FTS table and k * (number_of_generated_questions + 1) rows in the vector table referencing to the same parent "generative" chunk id.
- During retrieval, top-k is retrieved from FTS table and top-k' is retrieved from vector table, where k' > k to account for the 1-many relationship in vector table. Construct "retrieval" chunk ranking for vector search by their best ranking from the k' retrieved vectors. Then, combine the vector ranking and FTS ranking using RRF to get a final ranking of "retrieval" chunks.
- [Test] Different strategy for constructing vector ranking of "retrieval" chunks - best only, sum of inverse ranking as score, etc.
- [Test] Use cross-encoder reranker to rerank the "retrieval" chunks. (latency and cost concern)
- Finally, the "retrieval" chunks are mapped back to "generative" chunks. "Generative" chunk ranking is determined by their respective highest ranked "retrieval" chunk.
- During generation, the top-m "generative" chunks are used as context for generation.
- **Important**: All chats only have access to documents they own, identified by chat_id.

### Table definitions:

documents:

- id: int (PK)
- location: text
- metadata: jsonb
- owner_id: text
- chat_id: UUID (FK to chat.chat_id) ON DELETE CASCADE
- <timestamp mixin>

generative_chunks:

- id: int (PK)
- document_id: int (FK to documents.id) ON DELETE CASCADE
- chunk_text: text
- chunk_index: int
- <timestamp mixin>

retrieval_chunks_fts:

- id: int (PK)
- generative_chunk_id: int (FK to generative_chunks.id) ON DELETE CASCADE
- chat_id: UUID (FK to chat.chat_id) ON DELETE CASCADE
- chunk_text: text
- contextual_anchor: text
- keywords_phrases: text[] (line separated)
- generated_questions: text[] (line separated)
- ts_vector: tsvector (computed column, weighted: keywords & key phrases: A, chunk_text: B, generated_questions: C, contextual anchor: D)
- <timestamp mixin>

index: GIN(ts_vector), chat_id

retrieval_chunks_vector:

- id: int (PK)
- retrieval_chunk_id: int (FK to retrieval_chunks_fts.id) ON DELETE CASCADE
- chat_id: UUID (FK to chat.chat_id) ON DELETE CASCADE
- source: text (either "chunk" or "generated_question")
- text: text
- embedding: vector
- <timestamp mixin>

index: chat_id, dynamically create hnsw(embedding, vector_l2_ops) on partial index where (chat_id = 'a') when the number of embeddings in chat 'a' grow past a threshold. Below the threshold, exact NN is used for similarity search.

### HNSW
- shuffle before indexing
- `ef_search = 512`
- `m = 18`
- `efConstruction = 512`


## Ablation Test Design

### FTS Tests

| Test ID | Chunk Text | Contextual Anchor | Generated Questions | Keywords/Phrases | FTS Weight                                 | Reranker | Result | Notes                |
| ------- | ---------- | ----------------- | ------------------- | ---------------- | ------------------------------------------ | -------- | ------ | -------------------- |
| F0      | O          | X                 | X                   | X                | -                                          | X        |        | Baseline: chunk only |
| F1      | O          | O                 | X                   | X                | Equal                                      | X        |        | Add anchor           |
| F2      | O          | X                 | O                   | X                | Equal                                      | X        |        | Add questions        |
| F3      | O          | X                 | X                   | O                | Equal                                      | X        |        | Add keywords         |
| F4      | O          | O                 | O                   | O                | Keywords:A, Chunk:B, Questions:C, Anchor:D | X        |        | Full FTS weighted    |
| F5      | O          | O                 | O                   | O                | Keywords:A, Chunk:B, Questions:C, Anchor:D | O        |        | Full FTS + reranker  |

### Vector Search Tests

| Test ID | Chunk Text | Contextual Anchor | Generated Questions | Ranking Strategy | Reranker | Result                       | Notes                  |
| ------- | ---------- | ----------------- | ------------------- | ---------------- | -------- | ---------------------------- | ---------------------- |
| V0      | O          | X                 | X                   | Best only        | X        | accuracy: 0.02, score: -0.92 | Baseline: chunk only   |
| V1      | O          | O                 | X                   | Best only        | X        |                              | Add anchor             |
| V2      | O          | X                 | O                   | Best only        | X        |                              | Add questions          |
| V3      | O          | O                 | O                   | Best only        | X        |                              | Full vector            |
| V4      | O          | O                 | O                   | Sum inverse rank | X        |                              | Test ranking strategy  |
| V5      | O          | O                 | O                   | Best only        | O        |                              | Full vector + reranker |

**Metrics**: Retrieval accuracy (MRR, Recall@k), end-to-end answer quality, latency, cost per query

### HNSW Threshold Test

Compare Exact NN vs HNSW query latency across vector counts (100, 200, ..., 10000) to determine optimal threshold. Plot latency vs vector count for both methods.

**Metrics**: Query latency, memory usage, index build time

### Partial Index Performance Tests

| Test ID | Total Chats | % Large Chats | Result | Notes    |
| ------- | ----------- | ------------- | ------ | -------- |
| I0      | 1000        | 2%            |        | Baseline |
| I1      | 1000        | 5%            |        |          |
| I2      | 1000        | 10%           |        |          |
| I3      | 1000        | 20%           |        |          |
| I4      | 10000       | 2%            |        |          |
| I5      | 10000       | 5%            |        |          |
| I6      | 10000       | 10%           |        |          |
| I7      | 10000       | 20%           |        |          |
| I8      | 50000       | 2%            |        |          |
| I9      | 50000       | 5%            |        |          |
| I10     | 50000       | 10%           |        |          |
| I11     | 50000       | 20%           |        |          |

**Metrics**: Small chat query latency, large chat query latency, memory usage, index creation overhead
