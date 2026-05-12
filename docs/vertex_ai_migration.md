# Vertex AI Migration Guide

## 1. Overview

The whole project is designed to be swappable. Every major component sits behind an abstract base class, so migrating to production = write a new implementation of the same interface. RAGEngine, StrategyA, StrategyB, HyDERetriever — none of them need to change.

| Component | Base class | Local version | Production target |
|-----------|-----------|--------------|-------------------|
| Embeddings | BaseEmbedder | LocalEmbedder | textembedding-gecko@003 |
| Vector store | BaseVectorStore | FAISSVectorStore | Vertex AI Vector Search |
| Generative model | (duck-typed) | MockGenerativeModel | Gemini 1.5 Pro |
| Reranker | (duck-typed) | CrossEncoderReranker | Vertex AI Ranking API |

## 2. Embeddings

**Local:**
```python
from src.embeddings.local_embedder import LocalEmbedder
embedder = LocalEmbedder()  # all-MiniLM-L6-v2, 384d
```

**Production:**
```python
from vertexai.language_models import TextEmbeddingModel
model = TextEmbeddingModel.from_pretrained("textembedding-gecko@003")
embeddings = model.get_embeddings(texts)
vectors = [e.values for e in embeddings]
```

To keep the same interface, wrap it in a class:

```python
class VertexEmbedder(BaseEmbedder):
    def __init__(self, model_name="textembedding-gecko@003"):
        self._model = TextEmbeddingModel.from_pretrained(model_name)

    def embed(self, texts):
        embeddings = self._model.get_embeddings(texts)
        matrix = np.array([e.values for e in embeddings], dtype=np.float32)
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        return matrix / norms  # normalize for cosine

    def embed_query(self, query):
        return self.embed([query])[0]
```

Drop this into RAGEngine in place of LocalEmbedder, nothing else changes.

Note: gecko outputs 768-dim vectors vs MiniLM's 384. Update `EMBEDDING_DIM` in config.py and rebuild the index.

## 3. Vector Store

Going from FAISS to Vertex AI Vector Search (Matching Engine). Four steps:

**Create the index:**
```python
from google.cloud import aiplatform
aiplatform.init(project=PROJECT_ID, location=LOCATION)

index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
    display_name="rag-benchmark-index",
    dimensions=768,
    approximate_neighbors_count=150,
    distance_measure_type="DOT_PRODUCT_DISTANCE",
    leaf_node_embedding_count=1000,
    leaf_nodes_to_search_percent=10,
)
```

**Create an endpoint:**
```python
endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
    display_name="rag-benchmark-endpoint",
    public_endpoint_enabled=True,
)
```

**Deploy:**
```python
endpoint.deploy_index(
    index=index,
    deployed_index_id="rag_benchmark_deployed",
    display_name="rag-benchmark-deployed",
    machine_type="e2-standard-2",
    min_replica_count=1,
    max_replica_count=2,
)
```

**Query:**
```python
response = endpoint.find_neighbors(
    deployed_index_id="rag_benchmark_deployed",
    queries=[query_vector.tolist()],
    num_neighbors=top_k,
)
```

Same idea as FAISS — wrap it in a `VertexVectorStore(BaseVectorStore)` class that implements `add()`, `search()`, `delete()`, `count()`.

## 4. Generative Model

This is the easiest swap because MockGenerativeModel was designed to match the real API:

**Local:**
```python
model = MockGenerativeModel()
response = model.generate_content(prompt)
print(response.text)
```

**Production:**
```python
import vertexai
from vertexai.generative_models import GenerativeModel

vertexai.init(project=PROJECT_ID, location=LOCATION)
model = GenerativeModel("gemini-1.5-pro")
response = model.generate_content(prompt)
print(response.text)
```

Same interface — `.generate_content(prompt)` returns an object with `.text`. HyDERetriever and QueryExpander don't need any changes, just pass them the real model instead of the mock.

## 5. Reranker

**Local:**
```python
reranker = CrossEncoderReranker()  # ms-marco-MiniLM
```

**Production — Vertex AI Ranking API:**
```python
from google.cloud import discoveryengine_v1alpha as discoveryengine

client = discoveryengine.RankServiceClient()
request = discoveryengine.RankRequest(
    ranking_config=f"projects/{PROJECT_ID}/locations/{LOCATION}/rankingConfigs/default_ranking_config",
    model="semantic-ranker-512@latest",
    top_n=top_k,
    query=query,
    records=[
        discoveryengine.RankingRecord(id=chunk_id, content=text)
        for chunk_id, text in candidates
    ],
)
response = client.rank(request=request)
```

The Vertex ranker uses a bigger model so it should be more accurate, but adds a network round trip.

## 6. Auth

**Local dev:**
```bash
gcloud auth application-default login
```

That's it. Stores creds locally, all the google cloud client libs pick them up automatically.

**Production (GKE):** use Workload Identity Federation. Maps a k8s service account to a GCP service account, so pods get credentials automatically. No service account key files, no JSON keys to rotate.

```yaml
apiVersion: v1
kind: ServiceAccount
metadata:
  name: rag-pipeline
  annotations:
    iam.gke.io/gcp-service-account: rag-pipeline@PROJECT_ID.iam.gserviceaccount.com
```

## 7. Cost

| Component | Local | Vertex AI |
|-----------|-------|-----------|
| Embeddings | free (CPU) | ~$0.0001 / 1k chars |
| Vector search | free (FAISS) | ~$0.10 / 1M queries + node hours |
| Reranking | free (CPU) | ~$0.001 / 1k records |
| LLM (generation) | free (mock) | ~$0.002 / 1k chars |

For dev and benchmarking, the local stack is totally fine. Save the Vertex AI spend for production and final validation.

## 8. Production Notes

- **Agentic RAG** — query expansion becomes a LangGraph node that can loop if initial retrieval confidence is low. The agent decides whether to expand, rephrase, or try HyDE
- **Multi-turn** — inject last N conversation turns into the prompt so follow-up questions like "tell me more about the caching part" work without re-stating context
- **Monitoring** — track context_precision over time with Cloud Monitoring custom metrics. Alert if it drops below 0.6, probably means the corpus is stale or embeddings have drifted
- **A/B testing** — run Strategy A and B in parallel on live traffic, measure end-to-end answer quality not just retrieval metrics
- **Index refresh** — schedule nightly re-ingestion via Cloud Scheduler -> Cloud Run to keep vectors current
