package io.homecook.lucene;

import java.nio.file.Path;

public final class App {
    public static void main(String[] args) throws Exception {
        String indexDir = System.getenv().getOrDefault("LUCENE_INDEX_DIR", "./lucene-index");
        int port = Integer.parseInt(System.getenv().getOrDefault("LUCENE_HTTP_PORT", "9090"));
        LuceneIndex index = new LuceneIndex(Path.of(indexDir));
        Runtime.getRuntime().addShutdownHook(new Thread(() -> {
            try {
                index.close();
            } catch (Exception ignored) {
                // best-effort on SIGTERM
            }
        }));
        // HttpApi.start returns while Javalin keeps serving; do not wrap `index` in
        // try-with-resources or it would close the IndexWriter before any request is handled.
        HttpApi.start(index, port);
    }
}
