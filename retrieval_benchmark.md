# Retrieval Benchmark Report

## Executive Summary

This benchmark evaluated **3 queries** across two retrieval strategies.

**Average latency** — Strategy A: 46.2 ms avg; Strategy B: 433.5 ms avg; Strategy HYDE: 284.9 ms avg.

**Average MRR** — Strategy A: 1.0000; Strategy B: 1.0000; Strategy HYDE: 1.0000.

## Query: *How does the system handle peak load?*

### Strategy A

- **Latency:** 47.61 ms

|   Rank | Chunk ID       | Source Doc   |   Score | Text Preview                                                                      |
|--------|----------------|--------------|---------|-----------------------------------------------------------------------------------|
|      1 | doc_02_chunk_3 | doc_02       |  0.6    | before anticipated demand surges Cool-down periods prevent thrashing by enforcin… |
|      2 | doc_01_chunk_1 | doc_01       |  0.5469 | low latency during demand surges When peak load exceeds the capacity of a single… |
|      3 | doc_02_chunk_4 | doc_02       |  0.438  | ween consecutive scaling actions Well-tuned policies strike a balance between co… |

### Strategy B

- **Latency:** 458.15 ms
- **Expanded query:** traffic surge, demand spike, high throughput, capacity planning, horizontal scaling

|   Rank | Chunk ID       | Source Doc   |    Score | Text Preview                                                                      |
|--------|----------------|--------------|----------|-----------------------------------------------------------------------------------|
|      1 | doc_01_chunk_1 | doc_01       |   0.6325 | low latency during demand surges When peak load exceeds the capacity of a single… |
|      2 | doc_02_chunk_3 | doc_02       | -11.0155 | before anticipated demand surges Cool-down periods prevent thrashing by enforcin… |
|      3 | doc_10_chunk_0 | doc_10       | -10.7069 | Effective monitoring during traffic spikes relies on real-time dashboards that s… |

### Strategy HYDE

- **Latency:** 331.80 ms
- **Expanded query:** traffic surge, demand spike, high throughput, capacity planning, horizontal scaling

|   Rank | Chunk ID       | Source Doc   |   Score | Text Preview                                                                      |
|--------|----------------|--------------|---------|-----------------------------------------------------------------------------------|
|      1 | doc_01_chunk_1 | doc_01       |  0.6325 | low latency during demand surges When peak load exceeds the capacity of a single… |
|      2 | doc_03_chunk_4 | doc_03       | -9.7893 | even in multi-region deployments Throttling acts as a safety valve that graceful… |
|      3 | doc_01_chunk_4 | doc_01       | -9.4622 | on crosses predefined thresholds The elasticity of horizontal expansion allows t… |

#### Metrics Comparison

| Strategy   | Hit@K   |   MRR |   Ctx Precision |   Ctx Recall |   Faithfulness* |   Answer Rel.* |
|------------|---------|-------|-----------------|--------------|-----------------|----------------|
| A          | ✓       |     1 |          1      |         0.5  |            0.85 |            0.8 |
| B          | ✓       |     1 |          1      |         0.75 |            0.85 |            0.8 |
| HYDE       | ✓       |     1 |          0.6667 |         0.25 |            0.85 |            0.8 |

*\* Mocked values — in production, computed via RAGAS with LLM-generated answers.*

#### Analysis

Strategy A returned the most relevant document at rank 1 with context precision 1.00 and recall 0.50 (latency 47.6 ms). Strategy B returned the most relevant document at rank 1 with context precision 1.00 and recall 0.75 (latency 458.1 ms). Strategy HYDE returned the most relevant document at rank 1 with context precision 0.67 and recall 0.25 (latency 331.8 ms).

## Query: *What strategies are used for caching?*

### Strategy A

- **Latency:** 51.32 ms

|   Rank | Chunk ID       | Source Doc   |   Score | Text Preview                                                                      |
|--------|----------------|--------------|---------|-----------------------------------------------------------------------------------|
|      1 | doc_04_chunk_0 | doc_04       |  0.5776 | Distributed caches such as Redis and Memcached store frequently accessed data in… |
|      2 | doc_04_chunk_1 | doc_04       |  0.5703 | ueries from the primary database A cache-aside pattern lets the application chec… |
|      3 | doc_04_chunk_2 | doc_04       |  0.5058 | ult back for subsequent requests Write-through and write-behind strategies push … |

### Strategy B

- **Latency:** 434.96 ms
- **Expanded query:** distributed cache, Redis, Memcached, cache-aside, write-through, TTL expiration

|   Rank | Chunk ID       | Source Doc   |    Score | Text Preview                                                                      |
|--------|----------------|--------------|----------|-----------------------------------------------------------------------------------|
|      1 | doc_04_chunk_4 | doc_04       | -10.6048 | ribution when the cluster scales Time-to-live expiration policies and active inv… |
|      2 | doc_09_chunk_2 | doc_09       | -11.229  | heterogeneous processing speeds IP-hash deterministically maps a client address … |
|      3 | doc_04_chunk_0 | doc_04       | -10.9487 | Distributed caches such as Redis and Memcached store frequently accessed data in… |

### Strategy HYDE

- **Latency:** 260.15 ms
- **Expanded query:** distributed cache, Redis, Memcached, cache-aside, write-through, TTL expiration

|   Rank | Chunk ID       | Source Doc   |    Score | Text Preview                                                                      |
|--------|----------------|--------------|----------|-----------------------------------------------------------------------------------|
|      1 | doc_04_chunk_2 | doc_04       |  -9.0048 | ult back for subsequent requests Write-through and write-behind strategies push … |
|      2 | doc_04_chunk_3 | doc_04       | -10.5433 | cy for stronger read consistency Consistent hashing distributes keys evenly acro… |
|      3 | doc_08_chunk_1 | doc_08       | -10.5364 | ation round-trip for every query Under high concurrency, an undersized pool forc… |

#### Metrics Comparison

| Strategy   | Hit@K   |   MRR |   Ctx Precision |   Ctx Recall |   Faithfulness* |   Answer Rel.* |
|------------|---------|-------|-----------------|--------------|-----------------|----------------|
| A          | ✓       |     1 |          1      |          0.5 |            0.85 |            0.8 |
| B          | ✓       |     1 |          0.6667 |          0.5 |            0.85 |            0.8 |
| HYDE       | ✓       |     1 |          0.6667 |          0.5 |            0.85 |            0.8 |

*\* Mocked values — in production, computed via RAGAS with LLM-generated answers.*

#### Analysis

Strategy A returned the most relevant document at rank 1 with context precision 1.00 and recall 0.50 (latency 51.3 ms). Strategy B returned the most relevant document at rank 1 with context precision 0.67 and recall 0.50 (latency 435.0 ms). Strategy HYDE returned the most relevant document at rank 1 with context precision 0.67 and recall 0.50 (latency 260.1 ms).

## Query: *How are failures detected and recovered?*

### Strategy A

- **Latency:** 39.65 ms

|   Rank | Chunk ID       | Source Doc   |   Score | Text Preview                                                                      |
|--------|----------------|--------------|---------|-----------------------------------------------------------------------------------|
|      1 | doc_05_chunk_0 | doc_05       |  0.3798 | The circuit breaker pattern monitors outbound calls to a dependency and trips op… |
|      2 | doc_05_chunk_2 | doc_05       |  0.3622 | eing overwhelmed by retry storms After a configurable timeout the breaker transi… |
|      3 | doc_05_chunk_3 | doc_05       |  0.3471 | e dependency has regained health Successful probes reset the breaker to the clos… |

### Strategy B

- **Latency:** 407.35 ms
- **Expanded query:** fault tolerance, circuit breaker, failover, degradation, resilience, recovery

|   Rank | Chunk ID       | Source Doc   |    Score | Text Preview                                                                      |
|--------|----------------|--------------|----------|-----------------------------------------------------------------------------------|
|      1 | doc_05_chunk_0 | doc_05       |  -7.1859 | The circuit breaker pattern monitors outbound calls to a dependency and trips op… |
|      2 | doc_10_chunk_2 | doc_10       | -10.759  | tching genuine degradation early Distributed tracing with OpenTelemetry correlat… |
|      3 | doc_05_chunk_4 | doc_05       |  -8.2292 | and restore normal traffic flow Libraries like Hystrix, Resilience4j, and Polly … |

### Strategy HYDE

- **Latency:** 262.87 ms
- **Expanded query:** fault tolerance, circuit breaker, failover, degradation, resilience, recovery

|   Rank | Chunk ID       | Source Doc   |   Score | Text Preview                                                                      |
|--------|----------------|--------------|---------|-----------------------------------------------------------------------------------|
|      1 | doc_05_chunk_0 | doc_05       | -7.1859 | The circuit breaker pattern monitors outbound calls to a dependency and trips op… |
|      2 | doc_05_chunk_1 | doc_05       | -8.933  | equests with a fallback response In the open state, no traffic is forwarded to t… |
|      3 | doc_05_chunk_4 | doc_05       | -8.2292 | and restore normal traffic flow Libraries like Hystrix, Resilience4j, and Polly … |

#### Metrics Comparison

| Strategy   | Hit@K   |   MRR |   Ctx Precision |   Ctx Recall |   Faithfulness* |   Answer Rel.* |
|------------|---------|-------|-----------------|--------------|-----------------|----------------|
| A          | ✓       |     1 |               1 |       0.3333 |            0.85 |            0.8 |
| B          | ✓       |     1 |               1 |       0.6667 |            0.85 |            0.8 |
| HYDE       | ✓       |     1 |               1 |       0.3333 |            0.85 |            0.8 |

*\* Mocked values — in production, computed via RAGAS with LLM-generated answers.*

#### Analysis

Strategy A returned the most relevant document at rank 1 with context precision 1.00 and recall 0.33 (latency 39.6 ms). Strategy B returned the most relevant document at rank 1 with context precision 1.00 and recall 0.67 (latency 407.4 ms). Strategy HYDE returned the most relevant document at rank 1 with context precision 1.00 and recall 0.33 (latency 262.9 ms).

## Aggregate Metrics Comparison

| Strategy   | Hit Rate   |   Avg MRR |   Avg Ctx Precision |   Avg Ctx Recall |
|------------|------------|-----------|---------------------|------------------|
| A          | 100.00%    |         1 |              1      |           0.4444 |
| B          | 100.00%    |         1 |              0.8889 |           0.6389 |
| HYDE       | 100.00%    |         1 |              0.7778 |           0.3611 |
