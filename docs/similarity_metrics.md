# Similarity Metrics

## 1. Why Cosine Similarity

Formula:

$$
\cos(\theta) = \frac{A \cdot B}{\|A\| \times \|B\|}
$$

Range is [-1, 1]. Score of 1.0 means identical direction (same meaning), 0 means unrelated, -1 means opposite.

Why this was the right choice for this project:

- **Magnitude doesn't matter** — a short chunk and a long chunk about the same topic get the same score. Cosine only cares about direction, not length. This is important because our chunks vary in size depending on where the recursive splitter breaks
- **Works at 384 dimensions** — Euclidean distance starts to break down in high-dimensional spaces (everything becomes roughly equidistant). Cosine stays discriminative
- **Matches the training objective** — sentence-transformers (all-MiniLM-L6-v2) are trained to optimize cosine similarity. Using a different metric at search time would be a mismatch
- Basically every major vector DB defaults to cosine for text (Pinecone, Weaviate, Qdrant, Vertex AI)

## 2. How It's Implemented

We don't use FAISS's L2 index and convert to cosine. Instead there's a nice trick:

1. Normalize all vectors to unit length before adding them to FAISS
2. When ||v|| = 1 for all vectors, inner product simplifies:

```
A · B = ||A|| × ||B|| × cos(θ)
      = 1 × 1 × cos(θ)
      = cos(θ)
```

3. So `IndexFlatIP` (inner product) on unit vectors = exact cosine similarity
4. Bonus: avoids the `sqrt()` that IndexFlatL2 needs, so it's slightly faster

The normalization happens in LocalEmbedder:

```python
vector = vector / np.linalg.norm(vector)
```

Both doc embeddings and query embeddings get normalized. The `_l2_normalize` function in `local_embedder.py` handles this — it also guards against zero vectors to avoid division by zero.

## 3. Why Not the Alternatives

|  | Cosine | Euclidean (L2) | Dot Product |
|--|--------|---------------|-------------|
| Formula | (A·B) / (‖A‖×‖B‖) | √Σ(aᵢ - bᵢ)² | Σ(aᵢ × bᵢ) |
| Range | [-1, 1] | [0, ∞) | (-∞, +∞) |
| Magnitude-invariant? | yes | no | no |
| Good for | text embeddings | image features, spatial data | recs, pre-normalized vecs |
| FAISS index | IndexFlatIP + normalization | IndexFlatL2 | IndexFlatIP (no normalization) |

**Euclidean** — two chunks about the same topic but different lengths would have different L2 norms, so Euclidean would penalize the length difference even though they're semantically the same. Bad for text.

**Raw dot product** — without normalization, longer documents get bigger embeddings and score higher regardless of relevance. Creates a bias toward verbose chunks.

## 4. FAISS Index Types

| Index | Type | Speed | Accuracy | When to use |
|-------|------|-------|----------|-------------|
| IndexFlatIP | exact | slow | 100% | small corpus, <100k vectors. **this project** |
| IndexFlatL2 | exact | slow | 100% | when you want euclidean distance |
| IndexIVFFlat | approximate | fast | ~95% | 100k - 10M vectors |
| IndexIVFPQ | approximate | fastest | ~90% | 10M+, memory constrained |
| IndexHNSW | graph-based | fastest | ~98% | production, latency-critical |

We use IndexFlatIP because we have ~40-60 chunks total. Exact search is sub-millisecond at this scale. No point adding the complexity of approximate indexes until the corpus grows past 100k.

## 5. For Production

Vertex AI Vector Search uses dot product on normalized vectors internally (same idea as our IndexFlatIP trick). Their Tree-AH algorithm handles approximate nearest neighbor at scale.

Config would look something like:
```json
{
  "dimensions": 768,
  "approximateNeighborsCount": 150,
  "distanceMeasureType": "DOT_PRODUCT_DISTANCE",
  "algorithmConfig": {
    "treeAhConfig": {
      "leafNodeEmbeddingCount": 1000,
      "leafNodesToSearchPercent": 10
    }
  }
}
```

Note: dimensions would change from 384 to 768 when switching from all-MiniLM-L6-v2 to textembedding-gecko@003. See [vertex_ai_migration.md](vertex_ai_migration.md) for the full migration path.
