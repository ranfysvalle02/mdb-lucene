package io.homecook.lucene;

import java.util.List;
import java.util.Map;

import com.google.gson.Gson;
import com.google.gson.reflect.TypeToken;

import io.javalin.Javalin;
import io.javalin.http.HttpStatus;

public final class HttpApi {

    private static final Gson GSON = new Gson();

    private HttpApi() {}

    public static void start(LuceneIndex index, int port) {
        Javalin app = Javalin.create();

        app.get("/health", ctx -> ctx.result("ok"));

        app.post("/index", ctx -> {
            String raw = ctx.body();
            if (raw == null || raw.isBlank()) {
                ctx.status(HttpStatus.BAD_REQUEST).result("empty body");
                return;
            }
            IndexDoc body = GSON.fromJson(raw, IndexDoc.class);
            if (body == null || body._id == null || body._id.isBlank()) {
                ctx.status(HttpStatus.BAD_REQUEST).result("missing _id");
                return;
            }
            index.upsert(body._id, body.text == null ? "" : body.text, body.embedding);
            ctx.status(HttpStatus.NO_CONTENT);
        });

        app.post("/bulk", ctx -> {
            String raw = ctx.body();
            if (raw == null || raw.isBlank()) {
                ctx.status(HttpStatus.BAD_REQUEST).result("empty body");
                return;
            }
            java.lang.reflect.Type t = new TypeToken<List<Map<String, Object>>>() {}.getType();
            List<Map<String, Object>> docs = GSON.fromJson(raw, t);
            if (docs == null || docs.isEmpty()) {
                ctx.status(HttpStatus.BAD_REQUEST).result("empty array");
                return;
            }
            int n = index.bulkUpsert(docs);
            ctx.contentType("application/json").result(GSON.toJson(Map.of("indexed", n)));
        });

        app.delete("/index/{id}", ctx -> {
            String id = ctx.pathParam("id");
            index.delete(id);
            ctx.status(HttpStatus.NO_CONTENT);
        });

        app.get("/search", ctx -> {
            String q = ctx.queryParam("q");
            int k = 10;
            String kStr = ctx.queryParam("k");
            if (kStr != null && !kStr.isBlank()) {
                try {
                    k = Integer.parseInt(kStr);
                } catch (NumberFormatException ignored) {
                    ctx.status(HttpStatus.BAD_REQUEST).result(GSON.toJson(Map.of("error", "invalid k")));
                    return;
                }
            }
            if (q == null || q.isBlank()) {
                ctx.status(HttpStatus.BAD_REQUEST).result(GSON.toJson(Map.of("error", "missing q")));
                return;
            }
            String padStr = ctx.queryParam("pad");
            boolean pad = padStr == null || !padStr.equalsIgnoreCase("false");
            List<Map<String, Object>> hits = index.search(q, Math.max(1, Math.min(k, 100)), pad);
            ctx.contentType("application/json").result(GSON.toJson(hits));
        });

        // POST /vector — native Lucene HNSW kNN over the "embedding" field. Same primitive Atlas
        // $vectorSearch is built on; we surface it so the demo can prove the two are equivalent.
        // Body: {"vector": [...floats...], "k": 10}
        app.post("/vector", ctx -> {
            String raw = ctx.body();
            if (raw == null || raw.isBlank()) {
                ctx.status(HttpStatus.BAD_REQUEST).result(GSON.toJson(Map.of("error", "empty body")));
                return;
            }
            VectorQuery body = GSON.fromJson(raw, VectorQuery.class);
            if (body == null || body.vector == null || body.vector.length == 0) {
                ctx.status(HttpStatus.BAD_REQUEST).result(GSON.toJson(Map.of("error", "missing vector")));
                return;
            }
            int k = body.k == null ? 10 : body.k;
            k = Math.max(1, Math.min(k, 100));
            List<Map<String, Object>> hits = index.vectorSearch(body.vector, k);
            ctx.contentType("application/json").result(GSON.toJson(hits));
        });

        // POST /hybrid — score-level hybrid: one Lucene BooleanQuery combining a parsed BM25
        // query and a KnnFloatVectorQuery as SHOULD clauses, each wrapped in a BoostQuery with
        // caller-supplied weights. The capability Solr (function_score / bf / bq) and OpenSearch
        // (hybrid query) expose at the engine level, and that Atlas $rankFusion (rank-level RRF)
        // does NOT — the demo shows it runs in 30 lines of Java against the same Lucene primitive
        // that mongot itself runs.
        // Body: {"q": "...", "vector": [...floats...], "k": 10, "bm25_weight": 0.6, "vec_weight": 0.4}
        app.post("/hybrid", ctx -> {
            String raw = ctx.body();
            if (raw == null || raw.isBlank()) {
                ctx.status(HttpStatus.BAD_REQUEST).result(GSON.toJson(Map.of("error", "empty body")));
                return;
            }
            HybridQuery body = GSON.fromJson(raw, HybridQuery.class);
            if (body == null) {
                ctx.status(HttpStatus.BAD_REQUEST).result(GSON.toJson(Map.of("error", "bad body")));
                return;
            }
            boolean haveText = body.q != null && !body.q.isBlank();
            boolean haveVec = body.vector != null && body.vector.length > 0;
            if (!haveText && !haveVec) {
                ctx.status(HttpStatus.BAD_REQUEST).result(GSON.toJson(Map.of("error", "missing q and vector")));
                return;
            }
            int k = body.k == null ? 10 : body.k;
            k = Math.max(1, Math.min(k, 100));
            float bm25Weight = body.bm25_weight == null ? 0.5f : body.bm25_weight;
            float vecWeight = body.vec_weight == null ? 0.5f : body.vec_weight;
            List<Map<String, Object>> hits = index.hybridSearch(body.q, body.vector, bm25Weight, vecWeight, k);
            ctx.contentType("application/json").result(GSON.toJson(Map.of(
                    "hits", hits,
                    "bm25_weight", bm25Weight,
                    "vec_weight", vecWeight,
                    "k", k)));
        });

        app.start(port);
    }

    @SuppressWarnings("unused")
    private static final class IndexDoc {
        String _id;
        String text;
        float[] embedding;
    }

    @SuppressWarnings("unused")
    private static final class VectorQuery {
        float[] vector;
        Integer k;
    }

    @SuppressWarnings("unused")
    private static final class HybridQuery {
        String q;
        float[] vector;
        Integer k;
        Float bm25_weight;
        Float vec_weight;
    }
}
