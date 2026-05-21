"""
FastAPI UI for the Atlas Local + Lucene demo. Auto-seeds a movies corpus on startup,
then exposes a single search endpoint that returns BM25 + $vectorSearch + RRF.
"""

from __future__ import annotations

import os
import threading
import time
from contextlib import asynccontextmanager
from typing import Any

import httpx
from fastapi import FastAPI, Query
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.templating import Jinja2Templates
from fastapi.requests import Request
from pymongo import MongoClient
from sentence_transformers import SentenceTransformer

MONGO_URI = os.environ.get("MONGO_URI", "mongodb://mongodb:27017/?directConnection=true")
LUCENE_URL = os.environ.get("LUCENE_URL", "http://lucene-search:9090").rstrip("/")
DB_NAME = os.environ.get("DEMO_DB", "demo")
COLL_NAME = os.environ.get("DEMO_COLLECTION", "movies")
EMBED_MODEL = os.environ.get("EMBED_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

SAMPLE_MOVIES: list[dict[str, Any]] = [
    {"_id": "mv01", "title": "Galaxy Defenders", "genre": "scifi", "decade": 2010, "runtime_min": 128, "in_catalog": True,
     "plot": "A ragtag crew must repel an alien invasion and save Earth using an ancient weapon."},
    {"_id": "mv02", "title": "The Last Lighthouse", "genre": "thriller", "decade": 2020, "runtime_min": 102, "in_catalog": True,
     "plot": "A lonely keeper discovers a signal that predicts storms—and something worse beneath the waves."},
    {"_id": "mv03", "title": "Paper Hearts", "genre": "drama", "decade": 2010, "runtime_min": 96, "in_catalog": True,
     "plot": "Two rival journalists fall in love while investigating a political scandal in a small town."},
    {"_id": "mv04", "title": "Iron Circuit", "genre": "scifi", "decade": 2020, "runtime_min": 134, "in_catalog": True,
     "plot": "A retired engineer returns to the factory floor to stop a rogue AI from automating war."},
    {"_id": "mv05", "title": "Desert Run", "genre": "thriller", "decade": 2010, "runtime_min": 110, "in_catalog": True,
     "plot": "A courier crosses a wasteland pursued by bandits, carrying a vaccine in a broken-down truck."},
    {"_id": "mv06", "title": "Midnight Baker", "genre": "crime", "decade": 2020, "runtime_min": 99, "in_catalog": True,
     "plot": "A shy baker moonlights as a vigilante, leaving clues in frosting at crime scenes."},
    {"_id": "mv07", "title": "Echoes of Mars", "genre": "scifi", "decade": 2020, "runtime_min": 141, "in_catalog": True,
     "plot": "Colonists uncover fossils that rewrite human history—and awaken a dormant ecosystem."},
    {"_id": "mv08", "title": "The Quiet Heist", "genre": "crime", "decade": 2010, "runtime_min": 105, "in_catalog": True,
     "plot": "Thieves plan a silent robbery in a library where every book is a safe."},
    {"_id": "mv09", "title": "Skybound", "genre": "scifi", "decade": 2020, "runtime_min": 112, "in_catalog": True,
     "plot": "Teens build a glider from scrap to escape a walled city ruled by drones."},
    {"_id": "mv10", "title": "Cold Harbor", "genre": "crime", "decade": 2020, "runtime_min": 121, "in_catalog": True,
     "plot": "A detective solves a murder tied to smuggling routes under a frozen harbor."},
    {"_id": "mv11", "title": "Second Sun", "genre": "scifi", "decade": 2010, "runtime_min": 118, "in_catalog": True,
     "plot": "A physicist proves a second sun exists in orbit, triggering a global energy race."},
    {"_id": "mv12", "title": "Neon Samurai", "genre": "scifi", "decade": 2020, "runtime_min": 125, "in_catalog": True,
     "plot": "In a neon megacity, a samurai-for-hire hunts corrupt executives with a plasma blade."},
    {"_id": "mv13", "title": "The Orchard Thief", "genre": "crime", "decade": 2010, "runtime_min": 94, "in_catalog": True,
     "plot": "A thief returns stolen heirlooms to families, financed by stealing from criminals."},
    {"_id": "mv14", "title": "Gravity Well", "genre": "scifi", "decade": 2020, "runtime_min": 138, "in_catalog": True,
     "plot": "Astronauts trapped near a black hole must choose between time and survival."},
    {"_id": "mv15", "title": "Clockwork Carnival", "genre": "fantasy", "decade": 2010, "runtime_min": 107, "in_catalog": False,
     "plot": "A carnival appears overnight; its rides predict visitors' futures with uncanny accuracy."},
    {"_id": "mv16", "title": "River Ghost", "genre": "fantasy", "decade": 2020, "runtime_min": 101, "in_catalog": True,
     "plot": "A river guide helps a ghost finish a journey, learning why it cannot cross the rapids."},
    {"_id": "mv17", "title": "Code Green", "genre": "thriller", "decade": 2020, "runtime_min": 113, "in_catalog": True,
     "plot": "Hackers expose a carbon credit fraud that funds illegal mining in protected forests."},
    {"_id": "mv18", "title": "The Brass Key", "genre": "fantasy", "decade": 2010, "runtime_min": 122, "in_catalog": True,
     "plot": "Siblings inherit a key that opens doors in different centuries—but each visit has a cost."},
    {"_id": "mv19", "title": "Volcano Choir", "genre": "thriller", "decade": 2020, "runtime_min": 116, "in_catalog": True,
     "plot": "Scientists decode volcanic harmonics that warn of eruptions—and summon something listening."},
    {"_id": "mv20", "title": "Street Chess", "genre": "drama", "decade": 2020, "runtime_min": 92, "in_catalog": True,
     "plot": "A chess prodigy hustles in parks until a mysterious opponent offers a match for their life."},
]

# A sibling collection. The point isn't the reviews themselves — it's that they live alongside
# the movies in the same database, and the composable pipeline can `$lookup` them in one query.
SAMPLE_REVIEWS: list[dict[str, Any]] = [
    {"movie_id": "mv01", "reviewer": "ana",   "rating": 4, "text": "Solid invasion flick, the third act earns it."},
    {"movie_id": "mv01", "reviewer": "ben",   "rating": 3, "text": "Fun but the ancient-weapon trope is tired."},
    {"movie_id": "mv02", "reviewer": "ana",   "rating": 5, "text": "Slow burn, then dread. The keeper is haunting."},
    {"movie_id": "mv02", "reviewer": "ben",   "rating": 5, "text": "Best opening shot of the decade."},
    {"movie_id": "mv02", "reviewer": "cole",  "rating": 4, "text": "Loses a half-star for the muddled climax."},
    {"movie_id": "mv03", "reviewer": "dee",   "rating": 4, "text": "Sharp dialogue, surprising warmth."},
    {"movie_id": "mv04", "reviewer": "ana",   "rating": 4, "text": "The retired-engineer angle works better than it should."},
    {"movie_id": "mv04", "reviewer": "cole",  "rating": 5, "text": "Smart sci-fi about labor and automation."},
    {"movie_id": "mv05", "reviewer": "ben",   "rating": 3, "text": "Lean and tense, a little thin in the middle."},
    {"movie_id": "mv06", "reviewer": "dee",   "rating": 5, "text": "Tonal tightrope, sticks the landing."},
    {"movie_id": "mv06", "reviewer": "ana",   "rating": 4, "text": "Frosting clues are inspired."},
    {"movie_id": "mv07", "reviewer": "cole",  "rating": 4, "text": "Real ideas, actual stakes — rare for a Mars movie."},
    {"movie_id": "mv08", "reviewer": "ben",   "rating": 5, "text": "Heist as architectural puzzle. Beautifully quiet."},
    {"movie_id": "mv08", "reviewer": "ana",   "rating": 5, "text": "Every shot in the library is composed like a vault."},
    {"movie_id": "mv09", "reviewer": "dee",   "rating": 4, "text": "YA but earned. The drone city is unforgettable."},
    {"movie_id": "mv10", "reviewer": "cole",  "rating": 4, "text": "Tight little procedural, frostbitten and mean."},
    {"movie_id": "mv11", "reviewer": "ana",   "rating": 3, "text": "Premise outpaces the plot."},
    {"movie_id": "mv12", "reviewer": "ben",   "rating": 4, "text": "Style as substance; you'll either swoon or roll your eyes."},
    {"movie_id": "mv13", "reviewer": "dee",   "rating": 4, "text": "Sweet without being saccharine."},
    {"movie_id": "mv14", "reviewer": "cole",  "rating": 5, "text": "The math is wrong and it doesn't matter — the grief is right."},
    {"movie_id": "mv14", "reviewer": "ana",   "rating": 5, "text": "Beautiful, devastating, do not watch alone."},
    {"movie_id": "mv16", "reviewer": "ben",   "rating": 4, "text": "Quietly moving. The rapids metaphor lands."},
    {"movie_id": "mv17", "reviewer": "dee",   "rating": 4, "text": "Climate-thriller that actually understands its subject."},
    {"movie_id": "mv18", "reviewer": "cole",  "rating": 3, "text": "Fun premise, repetitive structure."},
    {"movie_id": "mv19", "reviewer": "ana",   "rating": 5, "text": "Volcanic harmonics is the best science-MacGuffin in years."},
    {"movie_id": "mv20", "reviewer": "ben",   "rating": 4, "text": "Park hustles done right. The opponent is genuinely scary."},
]

state: dict[str, Any] = {
    "ready": False,
    "indexed": 0,
    "reviews_indexed": 0,
    "embed_dim": None,
    "seed_error": None,
}


def rrf_fuse(rankings: list[list[str]], k: int = 60) -> list[tuple[str, float]]:
    scores: dict[str, float] = {}
    for ranking in rankings:
        for rank, doc_id in enumerate(ranking):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank + 1)
    return sorted(scores.items(), key=lambda kv: -kv[1])


def _wait_for_backends(client: httpx.Client, mongo: MongoClient, timeout_s: int = 120) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = client.get(f"{LUCENE_URL}/health", timeout=2.0)
            lucene_ok = r.status_code == 200 and r.text.strip() == "ok"
        except Exception:
            lucene_ok = False
        try:
            mongo[DB_NAME].list_collection_names()
            mongo_ok = True
        except Exception:
            mongo_ok = False
        if lucene_ok and mongo_ok:
            return
        time.sleep(1.5)
    raise RuntimeError(f"Backends not ready (lucene={LUCENE_URL}, mongo={MONGO_URI})")


VECTOR_INDEX_NAME = "vector_index"
REVIEWS_COLL = os.environ.get("DEMO_REVIEWS_COLLECTION", "reviews")


def _ensure_vector_index(coll, num_dims: int, timeout_s: int = 90) -> None:
    """Create (or recreate) the Atlas Vector Search index.

    Critically, the index includes `filter`-type fields alongside the `vector` field. Those
    `filter` fields are what enables Atlas to push a `filter` predicate *into* the HNSW graph
    traversal — pre-filter, not post-filter. That's a Layer-4 capability you can only get when
    the vectors live in the same engine that already knows how to read `genre`, `decade`, and
    `in_catalog`. Try doing this with a separate vector store and a separate operational DB.
    """
    desired = {
        "name": VECTOR_INDEX_NAME,
        "type": "vectorSearch",
        "definition": {
            "fields": [
                {"type": "vector", "numDimensions": num_dims, "path": "embedding", "similarity": "cosine"},
                {"type": "filter", "path": "genre"},
                {"type": "filter", "path": "decade"},
                {"type": "filter", "path": "in_catalog"},
            ]
        },
    }

    existing = {ix.get("name"): ix for ix in coll.list_search_indexes()}
    current = existing.get(VECTOR_INDEX_NAME)
    if current is None:
        coll.create_search_index(desired)
    else:
        # Drop+recreate if the index lacks our filter fields (e.g. older volume from before
        # the composable-pipeline refactor). Cheaper than a real schema migration in a demo.
        current_fields = {(f.get("type"), f.get("path"))
                          for f in (current.get("latestDefinition") or {}).get("fields", [])}
        desired_fields = {(f["type"], f["path"]) for f in desired["definition"]["fields"]}
        if not desired_fields.issubset(current_fields):
            try:
                coll.drop_search_index(VECTOR_INDEX_NAME)
            except Exception:
                pass
            # Wait for the drop to take effect before creating again.
            for _ in range(30):
                names = {ix.get("name") for ix in coll.list_search_indexes()}
                if VECTOR_INDEX_NAME not in names:
                    break
                time.sleep(1.0)
            coll.create_search_index(desired)

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        for ix in coll.list_search_indexes():
            if ix.get("name") == VECTOR_INDEX_NAME and ix.get("queryable") is True:
                return
        time.sleep(1.0)
    raise RuntimeError(f"Vector index '{VECTOR_INDEX_NAME}' not queryable after {timeout_s}s")


def _seed(mongo: MongoClient, http: httpx.Client, emb: SentenceTransformer) -> None:
    coll = mongo[DB_NAME][COLL_NAME]
    reviews = mongo[DB_NAME][REVIEWS_COLL]
    plots = [str(m["plot"]) for m in SAMPLE_MOVIES]
    vectors = emb.encode(plots, show_progress_bar=False).tolist()

    # Reseed when the schema changed (no `genre` field on existing docs) or when count is short.
    needs_reseed = (
        coll.estimated_document_count() < len(SAMPLE_MOVIES)
        or coll.find_one({"genre": {"$exists": False}}) is not None
    )
    if needs_reseed:
        coll.drop()
        coll.insert_many(
            [
                {**{k: v for k, v in m.items() if k != "_id"}, "_id": m["_id"], "embedding": vec}
                for m, vec in zip(SAMPLE_MOVIES, vectors)
            ]
        )

    if reviews.estimated_document_count() < len(SAMPLE_REVIEWS):
        reviews.drop()
        reviews.insert_many([{**r} for r in SAMPLE_REVIEWS])

    _ensure_vector_index(coll, num_dims=len(vectors[0]))

    # Always (re-)write the Lucene side. The persisted lucene-index volume might be from a previous
    # build that lacked the embedding field; Lucene's updateDocument fully replaces by _id, so this
    # cheaply makes the index forward-compatible with the new KnnFloatVectorField schema.
    r = http.post(
        f"{LUCENE_URL}/bulk",
        json=[
            {"_id": m["_id"], "text": m["plot"], "embedding": v}
            for m, v in zip(SAMPLE_MOVIES, vectors)
        ],
        timeout=60.0,
    )
    r.raise_for_status()
    state["indexed"] = coll.estimated_document_count() or len(SAMPLE_MOVIES)
    state["reviews_indexed"] = reviews.estimated_document_count() or len(SAMPLE_REVIEWS)


def _seed_with_retry(http: httpx.Client, mongo: MongoClient, emb: SentenceTransformer,
                     max_attempts: int = 60, sleep_s: float = 5.0) -> None:
    """Wait for backends + seed, retrying on transient failures.

    Runs in a background thread so uvicorn starts serving immediately. The /readyz
    endpoint returns 503 until this loop succeeds, so docker-compose `--wait`
    blocks until the demo is actually queryable. If a backend hiccups mid-startup
    (mongo's runner panics during replica-set init, lucene-search slow to bind,
    Atlas Local takes longer than the compose start_period to elect itself
    primary), this loop notices and retries instead of leaving the demo
    permanently broken with `seed_error` set.
    """
    for attempt in range(1, max_attempts + 1):
        try:
            _wait_for_backends(http, mongo, timeout_s=30)
            _seed(mongo, http, emb)
            state["ready"] = True
            state["seed_error"] = None
            return
        except Exception as e:
            state["seed_error"] = f"attempt {attempt}/{max_attempts}: {e!r}"
            time.sleep(sleep_s)
    # exhausted; seed_error stays set so /readyz keeps reporting the last failure


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.http = httpx.Client(timeout=30.0)
    app.state.mongo = MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000)
    app.state.emb = SentenceTransformer(EMBED_MODEL)
    state["embed_dim"] = int(app.state.emb.get_sentence_embedding_dimension())
    seed_thread = threading.Thread(
        target=_seed_with_retry,
        args=(app.state.http, app.state.mongo, app.state.emb),
        name="seed-with-retry",
        daemon=True,
    )
    seed_thread.start()
    try:
        yield
    finally:
        app.state.http.close()
        app.state.mongo.close()


app = FastAPI(title="mdb-lucene demo", lifespan=lifespan)
templates = Jinja2Templates(directory="templates")


@app.get("/healthz")
def healthz() -> dict[str, Any]:
    """Liveness + introspection. Always 200 if the process is alive — the JSON body
    tells you whether the seed succeeded. Use `/readyz` for a real readiness gate."""
    return {
        "ready": state["ready"],
        "indexed": state["indexed"],
        "reviews_indexed": state["reviews_indexed"],
        "embed_dim": state["embed_dim"],
        "seed_error": state["seed_error"],
        "mongo_uri": MONGO_URI,
        "lucene_url": LUCENE_URL,
    }


@app.get("/readyz")
def readyz() -> JSONResponse:
    """Readiness probe. Returns 200 only when the seed and vector index are both
    in place; returns 503 (with the failure reason) otherwise. The docker-compose
    healthcheck for this service hits `/readyz`, so `docker compose up --wait`
    blocks until the demo is actually queryable, not just until uvicorn binds."""
    if state["ready"]:
        return JSONResponse(
            {"ready": True, "indexed": state["indexed"], "reviews_indexed": state["reviews_indexed"]},
            status_code=200,
        )
    return JSONResponse(
        {"ready": False, "seed_error": state["seed_error"]},
        status_code=503,
    )


# The demo IS the proof. `/` renders templates/index.html — the engine-equivalence
# panel and the composable-pipeline panel side by side. The strategic frame those
# panels are one runnable instance of lives at https://demos.oblivio-company.com/why-mongodb/.
@app.get("/", response_class=HTMLResponse)
def index(request: Request) -> HTMLResponse:
    return templates.TemplateResponse(
        request,
        "index.html",
        {
            "indexed": state["indexed"],
            "reviews_indexed": state["reviews_indexed"],
            "embed_dim": state["embed_dim"],
            "ready": state["ready"],
            "seed_error": state["seed_error"],
            "sample_movies": SAMPLE_MOVIES,
            "default_query": "lonely keeper discovers signal danger",
        },
    )


def _lucene_search(http: httpx.Client, q: str, k: int) -> list[dict[str, Any]]:
    r = http.get(f"{LUCENE_URL}/search", params={"q": q, "k": k}, timeout=10.0)
    r.raise_for_status()
    return r.json()


def _lucene_vector(http: httpx.Client, vector: list[float], k: int) -> list[dict[str, Any]]:
    """Native Lucene HNSW kNN — same primitive that Atlas $vectorSearch sits on top of."""
    r = http.post(f"{LUCENE_URL}/vector", json={"vector": vector, "k": k}, timeout=10.0)
    r.raise_for_status()
    return r.json()


@app.get("/api/search")
def api_search(
    q: str | None = Query(None),
    bm25: str | None = Query(None),
    vector: str | None = Query(None),
    k: int = Query(8, ge=1, le=50),
) -> JSONResponse:
    """Run BM25, $vectorSearch, and RRF fusion.

    - Pass ?q=... and the same query is sent to both engines (default mode).
    - Pass ?bm25=...&vector=... to use different queries per engine (compare mode).
    """
    if not state["ready"]:
        return JSONResponse({"error": "service not ready", "details": state["seed_error"]}, status_code=503)

    bm25_q = (bm25 or q or "").strip()
    vec_q = (vector or q or "").strip()
    if not bm25_q or not vec_q:
        return JSONResponse({"error": "missing query: pass ?q=... or ?bm25=...&vector=..."}, status_code=400)

    http: httpx.Client = app.state.http
    mongo: MongoClient = app.state.mongo
    emb: SentenceTransformer = app.state.emb

    t0 = time.perf_counter()
    bm25_hits = _lucene_search(http, bm25_q, k)
    bm25_ms = (time.perf_counter() - t0) * 1000

    qvec = emb.encode([vec_q], show_progress_bar=False)[0].tolist()
    pipeline = [
        {"$vectorSearch": {
            "index": VECTOR_INDEX_NAME,
            "queryVector": qvec,
            "path": "embedding",
            "limit": k,
            "exact": True,
        }},
        {"$project": {"title": 1, "plot": 1, "_score": {"$meta": "vectorSearchScore"}}},
    ]
    t0 = time.perf_counter()
    vec_hits = list(mongo[DB_NAME][COLL_NAME].aggregate(pipeline))
    vec_ms = (time.perf_counter() - t0) * 1000

    # Same query embedding, fired at Lucene's native HNSW kNN. If mongodb's $vectorSearch and
    # Lucene HNSW return the same top-k for the same vector, that's the demo's punch line:
    # Atlas $vectorSearch is Lucene HNSW under the hood.
    t0 = time.perf_counter()
    lvec_hits = _lucene_vector(http, qvec, k)
    lvec_ms = (time.perf_counter() - t0) * 1000

    bm25_ids = [str(h["_id"]) for h in bm25_hits]
    vec_ids = [str(d["_id"]) for d in vec_hits]
    lvec_ids = [str(h["_id"]) for h in lvec_hits]

    # Lucene pads its result up to k with low-boost MatchAllDocsQuery so the BM25 and vector
    # columns line up 1:1 in the UI. Only TRULY matched BM25 hits should feed the RRF fusion -
    # otherwise filler docs would inherit unearned rank credit.
    bm25_matched_ids = [str(h["_id"]) for h in bm25_hits if bool(h.get("matched", True))]
    fused = rrf_fuse([bm25_matched_ids, vec_ids])[:k]

    meta = {m["_id"]: m for m in SAMPLE_MOVIES}
    bm25_set, vec_set = set(bm25_matched_ids), set(vec_ids)
    lvec_set = set(lvec_ids)
    bm25_rank = {doc_id: i + 1 for i, doc_id in enumerate(bm25_matched_ids)}
    vec_rank = {doc_id: i + 1 for i, doc_id in enumerate(vec_ids)}
    rrf_k_param = 60

    # Educational metric: how often does Lucene HNSW agree with mongodb $vectorSearch on the same
    # query embedding? They share the same primitive, so for exact/small datasets this should be 1.0.
    union = vec_set | lvec_set
    overlap_jaccard = (len(vec_set & lvec_set) / len(union)) if union else 1.0
    same_top1 = bool(vec_ids and lvec_ids and vec_ids[0] == lvec_ids[0])

    return JSONResponse({
        "bm25": {
            "query": bm25_q,
            "took_ms": round(bm25_ms, 2),
            "matched_count": len(bm25_matched_ids),
            "padded": len(bm25_hits) - len(bm25_matched_ids),
            "hits": [
                {
                    "rank": i + 1,
                    "_id": str(h["_id"]),
                    "title": meta.get(str(h["_id"]), {}).get("title", ""),
                    "plot": meta.get(str(h["_id"]), {}).get("plot", ""),
                    "score": float(h.get("score", 0)),
                    "matched": bool(h.get("matched", True)),
                }
                for i, h in enumerate(bm25_hits)
            ],
        },
        "vector": {
            "query": vec_q,
            "took_ms": round(vec_ms, 2),
            "hits": [
                {
                    "rank": i + 1,
                    "_id": str(d["_id"]),
                    "title": str(d.get("title", "")),
                    "plot": str(d.get("plot", meta.get(str(d["_id"]), {}).get("plot", ""))),
                    "score": float(d.get("_score", 0)),
                }
                for i, d in enumerate(vec_hits)
            ],
        },
        "lucene_vec": {
            "query": vec_q,
            "took_ms": round(lvec_ms, 2),
            "hits": [
                {
                    "rank": i + 1,
                    "_id": str(h["_id"]),
                    "title": meta.get(str(h["_id"]), {}).get("title", ""),
                    "plot": meta.get(str(h["_id"]), {}).get("plot", ""),
                    "score": float(h.get("score", 0)),
                    "agrees_with_mongodb": str(h["_id"]) in vec_set,
                }
                for i, h in enumerate(lvec_hits)
            ],
            "agreement": {
                "overlap_jaccard": round(overlap_jaccard, 3),
                "same_top1": same_top1,
                "overlap_count": len(vec_set & lvec_set),
                "k": k,
            },
        },
        "rrf": {
            "k_param": rrf_k_param,
            "hits": [
                {
                    "rank": i + 1,
                    "_id": doc_id,
                    "title": meta.get(doc_id, {}).get("title", ""),
                    "plot": meta.get(doc_id, {}).get("plot", ""),
                    "score": round(score, 6),
                    "from": [src for src, ok in (("BM25", doc_id in bm25_set), ("vec", doc_id in vec_set)) if ok],
                    "bm25_rank": bm25_rank.get(doc_id),
                    "vec_rank": vec_rank.get(doc_id),
                    "rrf_breakdown": {
                        "bm25_contrib": round(1.0 / (rrf_k_param + bm25_rank[doc_id]), 6) if doc_id in bm25_rank else 0,
                        "vec_contrib": round(1.0 / (rrf_k_param + vec_rank[doc_id]), 6) if doc_id in vec_rank else 0,
                    },
                }
                for i, (doc_id, score) in enumerate(fused)
            ],
        },
    })


# ---------- composable retrieval: the actual differentiation story ----------
#
# The other endpoint above proves the engine is portable. This one shows the part of MongoDB Atlas
# Vector Search that *isn't* portable, because it isn't even a property of the engine — it's a
# property of having the engine live inside the rest of your data and aggregation pipeline.
#
# A single $vectorSearch stage with a `filter` predicate (pre-filter pushdown into the HNSW
# graph traversal), $lookup against a sibling `reviews` collection, $addFields to compute an
# average rating, $match to require some minimum review evidence, and a final $project / $sort.
# All in one round trip, against one cluster, with one ops surface. No CDC pipeline keeping a
# separate vector store in sync; no application-side joins; no second auth model.

@app.get("/api/composable")
def api_composable(
    q: str = Query("astronauts grieving black hole"),
    genre: str | None = Query(None, description="filter pushed into HNSW (e.g. 'scifi')"),
    decade_gte: int | None = Query(None, description="filter pushed into HNSW (e.g. 2020)"),
    in_catalog: bool | None = Query(True, description="filter pushed into HNSW"),
    min_avg_rating: float | None = Query(None, description="post-$lookup filter on avg review rating"),
    k: int = Query(8, ge=1, le=50),
) -> JSONResponse:
    if not state["ready"]:
        return JSONResponse({"error": "service not ready", "details": state["seed_error"]}, status_code=503)

    http: httpx.Client = app.state.http
    mongo: MongoClient = app.state.mongo
    emb: SentenceTransformer = app.state.emb

    qvec = emb.encode([q], show_progress_bar=False)[0].tolist()

    # Build the filter predicate as a real Atlas vectorSearch filter (Lucene-syntax-equivalent
    # query operators on the indexed `filter`-type fields). Atlas pushes this *down* into the
    # HNSW traversal so we don't pay the cost of fetching neighbours we'd just throw away.
    vs_filter: dict[str, Any] = {}
    if genre:
        vs_filter["genre"] = {"$eq": genre}
    if decade_gte is not None:
        vs_filter["decade"] = {"$gte": decade_gte}
    if in_catalog is not None:
        vs_filter["in_catalog"] = {"$eq": in_catalog}

    vector_stage: dict[str, Any] = {
        "$vectorSearch": {
            "index": VECTOR_INDEX_NAME,
            "queryVector": qvec,
            "path": "embedding",
            "limit": k,
            "exact": True,
        }
    }
    if vs_filter:
        vector_stage["$vectorSearch"]["filter"] = vs_filter

    pipeline: list[dict[str, Any]] = [
        vector_stage,
        {"$project": {
            "title": 1, "plot": 1, "genre": 1, "decade": 1, "runtime_min": 1,
            "vector_score": {"$meta": "vectorSearchScore"},
        }},
        {"$lookup": {
            "from": REVIEWS_COLL,
            "localField": "_id",
            "foreignField": "movie_id",
            "as": "reviews",
        }},
        {"$addFields": {
            "review_count": {"$size": "$reviews"},
            "avg_rating": {"$cond": [
                {"$gt": [{"$size": "$reviews"}, 0]},
                {"$avg": "$reviews.rating"},
                None,
            ]},
        }},
    ]
    if min_avg_rating is not None:
        pipeline.append({"$match": {"avg_rating": {"$gte": min_avg_rating}}})
    pipeline += [
        {"$project": {
            "title": 1, "plot": 1, "genre": 1, "decade": 1, "runtime_min": 1,
            "vector_score": 1, "review_count": 1, "avg_rating": 1,
            "top_review": {"$first": "$reviews.text"},
        }},
        {"$sort": {"vector_score": -1}},
    ]

    t0 = time.perf_counter()
    results = list(mongo[DB_NAME][COLL_NAME].aggregate(pipeline))
    composable_ms = (time.perf_counter() - t0) * 1000

    # And here's the contrast the demo is built to make visible. With a separate vector store
    # (or raw Lucene without a database around it), the same workflow becomes 4 round trips
    # plus application-side glue. We DO it, end-to-end, so the timings are honest.
    t0 = time.perf_counter()
    lvec_resp = http.post(f"{LUCENE_URL}/vector", json={"vector": qvec, "k": k * 4}, timeout=10.0)
    lvec_resp.raise_for_status()
    lvec_hits_raw = lvec_resp.json()
    t_lucene_ms = (time.perf_counter() - t0) * 1000

    # Step 2: post-filter in app code (this is what destroys recall when filters are selective —
    # we asked for k*4 just to leave ourselves a chance, and even that doesn't help if the
    # filter is rare. With Atlas, the `filter` rides INSIDE the HNSW traversal, so you don't
    # over-fetch and you don't lose recall.)
    meta = {m["_id"]: m for m in SAMPLE_MOVIES}
    t0 = time.perf_counter()
    post_filtered = []
    for h in lvec_hits_raw:
        m = meta.get(str(h.get("_id"))) or {}
        if genre and m.get("genre") != genre:
            continue
        if decade_gte is not None and (m.get("decade") or 0) < decade_gte:
            continue
        if in_catalog is not None and bool(m.get("in_catalog")) != bool(in_catalog):
            continue
        post_filtered.append({"_id": str(h["_id"]), "score": float(h.get("score", 0)), **m})
        if len(post_filtered) >= k:
            break
    t_postfilter_ms = (time.perf_counter() - t0) * 1000

    # Step 3: app-side $lookup against the reviews collection. In a real "vector DB lives over
    # here, operational data lives over there" architecture, this is N HTTP calls or one batched
    # query against a separate system, plus the orchestration code. We just hit Mongo because we
    # have it; in production the whole point is that you wouldn't.
    reviews_coll = mongo[DB_NAME][REVIEWS_COLL]
    t0 = time.perf_counter()
    review_map: dict[str, list[dict[str, Any]]] = {}
    if post_filtered:
        ids = [d["_id"] for d in post_filtered]
        for r in reviews_coll.find({"movie_id": {"$in": ids}}):
            review_map.setdefault(r["movie_id"], []).append(r)
    enriched = []
    for d in post_filtered:
        rs = review_map.get(d["_id"], [])
        avg = (sum(r["rating"] for r in rs) / len(rs)) if rs else None
        if min_avg_rating is not None and (avg is None or avg < min_avg_rating):
            continue
        enriched.append({
            "_id": d["_id"], "title": d.get("title", ""), "plot": d.get("plot", ""),
            "genre": d.get("genre"), "decade": d.get("decade"), "runtime_min": d.get("runtime_min"),
            "vector_score": d["score"], "review_count": len(rs), "avg_rating": avg,
            "top_review": rs[0]["text"] if rs else None,
        })
    t_join_ms = (time.perf_counter() - t0) * 1000
    total_lucene_ms = t_lucene_ms + t_postfilter_ms + t_join_ms

    return JSONResponse({
        "query": q,
        "filter": vs_filter,
        "min_avg_rating": min_avg_rating,
        "k": k,
        "atlas": {
            "took_ms": round(composable_ms, 2),
            "round_trips": 1,
            "pipeline": _serialize_pipeline_for_display(pipeline),
            "hits": [
                {
                    "rank": i + 1,
                    "_id": str(d["_id"]),
                    "title": str(d.get("title", "")),
                    "plot": str(d.get("plot", "")),
                    "genre": d.get("genre"),
                    "decade": d.get("decade"),
                    "runtime_min": d.get("runtime_min"),
                    "vector_score": float(d.get("vector_score", 0)),
                    "review_count": int(d.get("review_count", 0)),
                    "avg_rating": (None if d.get("avg_rating") is None else round(float(d["avg_rating"]), 2)),
                    "top_review": d.get("top_review"),
                }
                for i, d in enumerate(results)
            ],
        },
        "raw_lucene_plus_glue": {
            "took_ms": round(total_lucene_ms, 2),
            "breakdown_ms": {
                "1_vector_call": round(t_lucene_ms, 2),
                "2_post_filter": round(t_postfilter_ms, 2),
                "3_app_side_join_and_aggregate": round(t_join_ms, 2),
            },
            "round_trips": 2,  # /vector, then reviews query
            "over_fetched": len(lvec_hits_raw),
            "kept_after_filter": len(post_filtered),
            "kept_after_min_rating": len(enriched),
            "hits": [
                {
                    "rank": i + 1,
                    "_id": d["_id"],
                    "title": d["title"],
                    "plot": d["plot"],
                    "genre": d.get("genre"),
                    "decade": d.get("decade"),
                    "runtime_min": d.get("runtime_min"),
                    "vector_score": d["vector_score"],
                    "review_count": d["review_count"],
                    "avg_rating": (None if d["avg_rating"] is None else round(d["avg_rating"], 2)),
                    "top_review": d["top_review"],
                }
                for i, d in enumerate(enriched)
            ],
        },
    })


def _serialize_pipeline_for_display(pipeline: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Strip the giant 384-d vector out so the UI can render the pipeline as readable JSON."""
    out: list[dict[str, Any]] = []
    for stage in pipeline:
        if "$vectorSearch" in stage:
            vs = dict(stage["$vectorSearch"])
            qv = vs.get("queryVector")
            if isinstance(qv, list):
                vs["queryVector"] = f"[…{len(qv)} floats…]"
            out.append({"$vectorSearch": vs})
        else:
            out.append(stage)
    return out
