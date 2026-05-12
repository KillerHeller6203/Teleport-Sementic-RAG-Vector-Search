"""
Corpus data module.

Defines the document corpus used for embedding and retrieval benchmarking.
Contains 10 technical paragraphs on distributed systems and infrastructure
engineering, plus ground-truth relevance judgments for evaluation queries.
"""

DOCUMENTS = [
    {
        "id": "doc_01",
        "title": "Horizontal Scaling Under Peak Load",
        "text": (
            "Horizontal scaling distributes incoming traffic across multiple "
            "server instances to maintain low latency during demand surges. "
            "When peak load exceeds the capacity of a single node, the "
            "orchestration layer provisions additional replicas behind the "
            "load balancer within seconds. Stateless service design is a "
            "prerequisite, because each replica must handle any request "
            "without relying on local session state. Container orchestrators "
            "such as Kubernetes track CPU and memory pressure per pod and "
            "emit scaling events when utilization crosses predefined "
            "thresholds. The elasticity of horizontal expansion allows the "
            "cluster to absorb sudden traffic spikes—like flash sales or "
            "viral events—without degrading response times for end users."
        ),
    },
    {
        "id": "doc_02",
        "title": "Auto-Scaling Policies and Triggers",
        "text": (
            "Auto-scaling policies define the rules that govern when and how "
            "compute resources are added or removed from a service fleet. "
            "Reactive triggers fire when real-time metrics—such as CPU "
            "utilization, request queue depth, or p99 latency—breach a "
            "configured threshold for a sustained window. Predictive "
            "auto-scaling leverages historical traffic patterns and time-"
            "series forecasting models to pre-provision capacity before "
            "anticipated demand surges. Cool-down periods prevent thrashing "
            "by enforcing a minimum interval between consecutive scaling "
            "actions. Well-tuned policies strike a balance between cost "
            "efficiency and availability, ensuring the system neither "
            "over-provisions idle resources nor under-provisions during "
            "critical traffic ramps."
        ),
    },
    {
        "id": "doc_03",
        "title": "Rate Limiting and Throttling Strategies",
        "text": (
            "Rate limiting controls the volume of requests a client can "
            "issue within a given time window, protecting backend services "
            "from abusive or runaway traffic. Token-bucket and sliding-"
            "window algorithms are the most widely adopted, each offering "
            "different burst-tolerance characteristics. When a client "
            "exceeds its allotted quota, the gateway returns an HTTP 429 "
            "response and includes a Retry-After header indicating when "
            "the client may resume. Distributed rate limiters synchronize "
            "counters across gateway nodes using an in-memory data store "
            "like Redis, ensuring consistent enforcement even in multi-"
            "region deployments. Throttling acts as a safety valve that "
            "gracefully degrades throughput rather than allowing cascading "
            "overloads to propagate downstream."
        ),
    },
    {
        "id": "doc_04",
        "title": "Distributed Caching Strategies",
        "text": (
            "Distributed caches such as Redis and Memcached store frequently "
            "accessed data in memory, drastically reducing read latency and "
            "offloading repetitive queries from the primary database. A "
            "cache-aside pattern lets the application check the cache first "
            "and fall back to the database only on a miss, writing the "
            "result back for subsequent requests. Write-through and write-"
            "behind strategies push updates into the cache at write time, "
            "trading additional write latency for stronger read consistency. "
            "Consistent hashing distributes keys evenly across cache nodes "
            "and minimizes key redistribution when the cluster scales. "
            "Time-to-live expiration policies and active invalidation hooks "
            "prevent stale data from persisting beyond an acceptable "
            "freshness window."
        ),
    },
    {
        "id": "doc_05",
        "title": "Circuit Breaker Pattern for Fault Tolerance",
        "text": (
            "The circuit breaker pattern monitors outbound calls to a "
            "dependency and trips open when the failure rate exceeds a "
            "threshold, immediately short-circuiting subsequent requests "
            "with a fallback response. In the open state, no traffic is "
            "forwarded to the degraded downstream, giving it time to "
            "recover without being overwhelmed by retry storms. After a "
            "configurable timeout the breaker transitions to half-open, "
            "allowing a limited probe of requests to test whether the "
            "dependency has regained health. Successful probes reset the "
            "breaker to the closed state and restore normal traffic flow. "
            "Libraries like Hystrix, Resilience4j, and Polly provide "
            "production-grade implementations with metrics dashboards "
            "that surface failure rates, state transitions, and fallback "
            "invocation counts in real time."
        ),
    },
    {
        "id": "doc_06",
        "title": "Vector Index Performance and ANN Algorithms",
        "text": (
            "Approximate nearest-neighbor algorithms trade a small amount "
            "of recall for orders-of-magnitude speedup over brute-force "
            "search in high-dimensional vector spaces. HNSW constructs a "
            "hierarchical navigable small-world graph that supports sub-"
            "millisecond queries even on collections exceeding ten million "
            "vectors. IVF-PQ partitions the vector space into Voronoi "
            "cells and applies product quantization to compress residuals, "
            "yielding compact indexes that fit entirely in RAM. Tuning "
            "parameters such as ef_search, nprobe, and the number of "
            "sub-quantizers directly influence the recall-latency trade-"
            "off. Benchmarking with realistic query distributions is "
            "essential, because synthetic uniform queries can mask "
            "performance cliffs that appear under skewed production "
            "workloads."
        ),
    },
    {
        "id": "doc_07",
        "title": "Embedding Model Latency and Batching",
        "text": (
            "Transformer-based embedding models incur non-trivial inference "
            "latency, especially when encoding long sequences on CPU-only "
            "hardware. Batching multiple texts into a single forward pass "
            "amortizes the fixed overhead of model loading and GPU kernel "
            "launches, increasing throughput by up to an order of magnitude. "
            "Dynamic batching collects incoming requests over a short "
            "accumulation window and dispatches them as a single padded "
            "tensor, balancing latency against utilization. Model "
            "distillation and ONNX Runtime optimization further reduce per-"
            "query inference time without significant loss in embedding "
            "quality. Monitoring p50 and p99 encode latencies is critical "
            "for sizing the embedding service and setting appropriate "
            "timeout budgets in the retrieval pipeline."
        ),
    },
    {
        "id": "doc_08",
        "title": "Database Connection Pooling Under High Concurrency",
        "text": (
            "Connection pooling maintains a reusable set of database "
            "connections that application threads borrow and return, "
            "eliminating the overhead of establishing a new TCP handshake "
            "and authentication round-trip for every query. Under high "
            "concurrency, an undersized pool forces incoming requests to "
            "queue, inflating tail latencies and risking cascading timeouts "
            "across dependent services. External poolers such as PgBouncer "
            "and ProxySQL sit between the application and the database, "
            "multiplexing thousands of client connections onto a smaller "
            "set of server connections. Idle-connection reaping and maximum-"
            "lifetime settings ensure that stale or leaked connections do "
            "not exhaust the pool. Proper pool sizing—guided by Little's "
            "Law and observed query durations—is one of the highest-leverage "
            "tuning actions for latency-sensitive workloads."
        ),
    },
    {
        "id": "doc_09",
        "title": "Load Balancer Algorithms",
        "text": (
            "Load balancers distribute inbound requests across a pool of "
            "backend servers using algorithms that optimize for different "
            "objectives. Round-robin cycles through servers sequentially "
            "and works well when instances are homogeneous and request "
            "costs are uniform. Least-connections routing directs each new "
            "request to the server with the fewest active connections, "
            "naturally adapting to heterogeneous processing speeds. IP-hash "
            "deterministically maps a client address to a specific backend, "
            "providing session affinity without requiring sticky-session "
            "cookies. Health checks continuously probe each backend and "
            "remove unhealthy instances from the rotation, ensuring that "
            "traffic is never forwarded to a server that cannot respond."
        ),
    },
    {
        "id": "doc_10",
        "title": "Monitoring and Alerting During Traffic Spikes",
        "text": (
            "Effective monitoring during traffic spikes relies on real-time "
            "dashboards that surface the four golden signals: latency, "
            "traffic volume, error rate, and resource saturation. Alerting "
            "rules should use multi-window burn-rate policies rather than "
            "static thresholds, reducing false positives while still "
            "catching genuine degradation early. Distributed tracing with "
            "OpenTelemetry correlates individual requests across service "
            "boundaries, enabling engineers to pinpoint bottlenecks that "
            "emerge only under load. Log aggregation pipelines funnel "
            "structured events into a central store where anomaly-detection "
            "models can flag unusual patterns automatically. Post-incident "
            "reviews that reference time-aligned metrics and traces are "
            "essential for hardening the system against future surges."
        ),
    },
]

GROUND_TRUTH = [
    {
        "query": "How does the system handle peak load?",
        "relevant_ids": ["doc_01", "doc_02", "doc_09", "doc_10"],
    },
    {
        "query": "What strategies are used for caching?",
        "relevant_ids": ["doc_04", "doc_03"],
    },
    {
        "query": "How are failures detected and recovered?",
        "relevant_ids": ["doc_05", "doc_09", "doc_10"],
    },
]
