# mdb-lucene

A side-by-side, runnable demo of:

1. **Engine equivalence** — Atlas `$vectorSearch` vs. raw Apache Lucene HNSW kNN over the
   same query embedding. Same primitive, same top-k, same scores (modulo float noise).
2. **Composable retrieval** — one `$vectorSearch` (with `filter` pushed *into* the HNSW
   traversal) + `$lookup` against a sibling `reviews` collection + `$addFields` +
   `$match` + `$sort`, all in a single round trip — vs. the same workflow built on a
   bare vector store with app-side post-filter and join.

Stack: MongoDB Atlas Local (real `$vectorSearch`, real search-index API), a ~200-line
Java/Lucene service exposing BM25 + HNSW over plain HTTP, and a FastAPI UI that wires
them together and renders both panels.

## Run it

```bash
cp .env.example .env       # optional; only needed to remap host ports
docker compose up --wait   # blocks until /readyz returns 200
open http://localhost:8088
```

First boot takes ~60–90s: pulling `mongodb-atlas-local`, building the Java service,
seeding 20 movies + 26 reviews, building the Atlas Vector Search index, and indexing
the same docs into Lucene. Subsequent boots are seconds.

Tear down:

```bash
docker compose down            # keep data volumes
docker compose down -v         # nuke everything
```

## Endpoints

| URL                                | What it returns                                                     |
| ---------------------------------- | ------------------------------------------------------------------- |
| `GET /`                            | The two-panel UI (engine equivalence + composable pipeline)         |
| `GET /healthz`                     | Liveness + introspection JSON (always 200)                          |
| `GET /readyz`                      | Readiness; 200 once seed + vector index are queryable, else 503     |
| `GET /api/search?q=...&k=8`        | BM25 (Lucene) + `$vectorSearch` + Lucene HNSW + RRF fusion          |
| `GET /api/composable?q=...&genre=` | One Atlas pipeline vs. raw-Lucene-plus-app-glue, with timings       |

The Lucene service speaks HTTP directly on `${LUCENE_PORT:-9090}`:

| URL                  | What it does                                       |
| -------------------- | -------------------------------------------------- |
| `GET /health`        | Liveness                                           |
| `POST /bulk`         | Upsert `[{_id, text, embedding[]}]`                |
| `POST /index`        | Upsert a single document                           |
| `DELETE /index/{id}` | Delete by `_id`                                    |
| `GET /search?q=&k=`  | BM25 (optionally padded to `k` with `?pad=false`)  |
| `POST /vector`       | HNSW kNN over `embedding` (`{"vector":[],"k":10}`) |

## Layout

```
.
├── docker-compose.yml           # mongodb (Atlas Local) + lucene-search + demo-ui
├── .env.example                 # host port overrides
└── services/
    ├── demo-ui/                 # FastAPI + Jinja UI; seeds both backends on startup
    │   ├── app.py
    │   ├── templates/index.html
    │   ├── requirements.txt
    │   └── Dockerfile
    └── lucene-search/           # Java 21 + Lucene 9 + Javalin; BM25 + HNSW over HTTP
        ├── src/main/java/io/homecook/lucene/
        ├── build.gradle.kts
        ├── settings.gradle.kts
        └── Dockerfile
```

## Configuration

Host ports (only override if `27018`, `9090`, or `8088` are taken):

```bash
WIRE_PORT=27018       # mongod -> host
LUCENE_PORT=9090      # lucene-search -> host
DEMO_UI_PORT=8088     # demo-ui     -> host
```

The container-to-container URIs (`MONGO_URI`, `LUCENE_URL`), database/collection names,
and embedding model are pinned in `docker-compose.yml` — change them there if needed.

## Notes

- The MongoDB `mongodb-data` and `mongodb-config` volumes are both persisted; without
  the latter, the Atlas Local runner regenerates the replica-set keyfile on every
  restart and the second `up` fails with "Unable to acquire security key[s]".
- The Atlas Vector Search index includes `filter`-type fields (`genre`, `decade`,
  `in_catalog`) so the composable panel can push predicates *into* the HNSW traversal,
  not post-filter the results.
- The Lucene `/search` response is padded with low-boost `MatchAllDocsQuery` filler so
  BM25 and `$vectorSearch` columns line up 1:1 in the UI; the `matched` flag on each
  hit distinguishes real BM25 matches from filler, and only real matches feed RRF.

## License

MIT — see [LICENSE](LICENSE).
