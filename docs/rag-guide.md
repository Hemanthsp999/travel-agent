# How the Travel Agent RAG Works

RAG (retrieval-augmented generation) lets the travel agent look up relevant knowledge and use it to answer a user. In this project, retrieval is a shared step in the main LangGraph orchestrator: it runs before the request is answered directly or routed to a hotel or flight subagent.

## 1. Knowledge sources

The retriever builds a set of documents from two sources:

- **FAQs and general policies** defined in `travel_agent/retrieval/knowledge.py`. These cover common questions such as how cancellations work and which requests need a live availability check.
- **Hotel and flight records** read from the mock API SQLite databases: `data/hotel.db` and `data/flight.db`, or the paths configured through the mock API settings. If a database is absent or not initialized, the retriever falls back to the corresponding mock seed data.

For each hotel or flight, the retriever also creates entity-specific rules and cancellation-policy documents. These are generated from mock records and policy maps in code. They are **illustrative mock policies**, not terms fetched from real hotels, airlines, or ticket bookings.

Document construction is in `travel_agent/retrieval/knowledge.py::_build_documents`; database loaders are `_hotel_rows` and `_flight_rows`.

## 2. FAISS index

The document text is converted to a vector by `_embed`. This implementation uses normalized hashed word and phrase features. It does **not** call a language model or an external embedding service to create vectors.

The vectors are inserted into a FAISS `IndexFlatIP` index. The index is held **in memory**, shared and cached for the lifetime of the running Python process. It is rebuilt when the process starts; it is not currently saved as a FAISS index file.

The implementation is in `KnowledgeRetriever` and `get_knowledge_retriever` in `travel_agent/retrieval/knowledge.py`. The `faiss-cpu` dependency is listed in `requirements.txt`.

## 3. Request and retrieval flow

The orchestrator observes the current user message, then runs the shared retrieval node before deciding where to route it:

1. The retriever checks whether the message looks like a FAQ/policy question or an operational request.
2. It creates a query vector and searches the FAISS index for relevant documents.
3. It considers domain clues (hotel, flight, or general) and explicit entity identifiers or names.
4. It returns ranked matches, source references, confidence information, formatted answer text when appropriate, and retrieval context.

The orchestrator integration is in `travel_agent/orchestrator/graph.py::_retrieve`; query classification and search are in `travel_agent/retrieval/knowledge.py::is_knowledge_query` and `KnowledgeRetriever.retrieve`.

### FAQ and policy questions

For a FAQ, rule, or cancellation-policy question, the main orchestrator answers from retrieved knowledge without calling a booking subagent. Answers are formatted with a heading and separate bullet points for distinct conditions.

### Operational requests

Requests that need live booking-system work—such as availability, prices, booking lookups, booking, or a cancellation action—continue to the relevant hotel or flight subagent. When retrieval has relevant context, the orchestrator includes it in the subagent task through `_subagent_context` in `travel_agent/orchestrator/graph.py`.

Retrieval provides supporting context; it does not replace the live API checks or the subagents' confirmation flow for changes to a booking.

## 4. Follow-up questions and refund eligibility

The orchestrator supplies the previous retrieval result to the next retrieval call in the same conversation. This allows a follow-up such as:

> “What is the cancellation policy for HT-1001?”  
> “Am I eligible for a refund?”

to retain the hotel context from the first question. Use the same `thread_id` for each message in a conversation so LangGraph can retrieve the saved state.

The agent does not infer a user's refund eligibility from general policy text alone. Eligibility may depend on the booking's rate or fare and the cancellation timing. If those details are not known, it explains the relevant policy and asks for the missing information instead of claiming the user is eligible or ineligible.

## 5. Try it

Run the focused retrieval tests:

```bash
.venv/bin/pytest -q tests/orchestrator/test_retrieval.py
```

Inspect retrieval results directly:

```bash
.venv/bin/python - <<'PY'
from travel_agent.retrieval import KnowledgeRetriever

retriever = KnowledgeRetriever()
result = retriever.retrieve("What is the cancellation policy for HT-1001?")

print("Vectors in FAISS:", retriever._index.ntotal)
print("Matches:", [match["document_id"] for match in result["matches"]])
print("\nAnswer:\n", result["answer"])
PY
```

For a follow-up query, pass the previous result to the next retrieval:

```python
policy = retriever.retrieve("What is the cancellation policy for HT-1001?")
eligibility = retriever.retrieve(
    "Am I eligible for a refund?",
    prior_knowledge=policy,
)
print(eligibility["answer"])
```

The same context handoff happens automatically through the orchestrator when requests share a `thread_id`.
