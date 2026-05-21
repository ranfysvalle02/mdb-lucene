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
}
