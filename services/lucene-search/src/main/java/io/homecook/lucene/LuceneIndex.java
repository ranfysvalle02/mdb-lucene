package io.homecook.lucene;

import java.io.IOException;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

import org.apache.lucene.analysis.standard.StandardAnalyzer;
import org.apache.lucene.document.Document;
import org.apache.lucene.document.Field;
import org.apache.lucene.document.KnnFloatVectorField;
import org.apache.lucene.document.StringField;
import org.apache.lucene.document.TextField;
import org.apache.lucene.index.DirectoryReader;
import org.apache.lucene.index.IndexWriter;
import org.apache.lucene.index.IndexWriterConfig;
import org.apache.lucene.index.Term;
import org.apache.lucene.index.VectorSimilarityFunction;
import org.apache.lucene.search.BooleanClause;
import org.apache.lucene.search.BooleanQuery;
import org.apache.lucene.search.BoostQuery;
import org.apache.lucene.search.Explanation;
import org.apache.lucene.search.IndexSearcher;
import org.apache.lucene.search.KnnFloatVectorQuery;
import org.apache.lucene.search.MatchAllDocsQuery;
import org.apache.lucene.search.Query;
import org.apache.lucene.search.ScoreDoc;
import org.apache.lucene.search.TopDocs;
import org.apache.lucene.store.Directory;
import org.apache.lucene.store.FSDirectory;

public final class LuceneIndex implements AutoCloseable {

    /** Vector field name. Matches the MongoDB "embedding" path so the two engines stay symmetric. */
    public static final String EMBEDDING_FIELD = "embedding";

    /**
     * Cosine matches Atlas $vectorSearch's default similarity. Lucene 9.x computes cosine on-the-fly
     * from non-normalised inputs, so callers do not have to pre-normalise the vector.
     */
    private static final VectorSimilarityFunction VECTOR_SIM = VectorSimilarityFunction.COSINE;

    private final Directory directory;
    private final IndexWriter writer;
    private DirectoryReader reader;

    public LuceneIndex(Path dir) throws IOException {
        this.directory = FSDirectory.open(dir);
        IndexWriterConfig cfg = new IndexWriterConfig(new StandardAnalyzer());
        cfg.setOpenMode(IndexWriterConfig.OpenMode.CREATE_OR_APPEND);
        this.writer = new IndexWriter(directory, cfg);
    }

    private synchronized IndexSearcher acquireSearcher() throws IOException {
        if (reader == null) {
            reader = DirectoryReader.open(writer);
        } else {
            DirectoryReader nr = DirectoryReader.openIfChanged(reader, writer);
            if (nr != null) {
                reader.close();
                reader = nr;
            }
        }
        return new IndexSearcher(reader);
    }

    public synchronized void upsert(String id, String text, float[] embedding) throws IOException {
        applyUpsert(id, text, embedding);
        writer.commit();
    }

    public synchronized int bulkUpsert(List<Map<String, Object>> docs) throws IOException {
        int n = 0;
        for (Map<String, Object> d : docs) {
            Object idObj = d.get("_id");
            if (idObj == null) {
                continue;
            }
            String id = idObj.toString();
            if (id.isBlank()) {
                continue;
            }
            String text = d.get("text") == null ? "" : d.get("text").toString();
            float[] vec = toFloatArray(d.get(EMBEDDING_FIELD));
            applyUpsert(id, text, vec);
            n++;
        }
        writer.commit();
        return n;
    }

    private void applyUpsert(String id, String text, float[] embedding) throws IOException {
        Document doc = new Document();
        doc.add(new StringField("_id", id, Field.Store.YES));
        doc.add(new TextField("text", text == null ? "" : text, Field.Store.NO));
        if (embedding != null && embedding.length > 0) {
            doc.add(new KnnFloatVectorField(EMBEDDING_FIELD, embedding, VECTOR_SIM));
        }
        writer.updateDocument(new Term("_id", id), doc);
    }

    public synchronized void delete(String id) throws IOException {
        writer.deleteDocuments(new Term("_id", id));
        writer.commit();
    }

    /** Anything below this score is from the MatchAllDocs filler, not a real BM25 match. */
    private static final float MATCH_THRESHOLD = 0.01f;

    /** Boost applied to MatchAllDocsQuery so docs with no query-token overlap still appear. */
    private static final float FILL_BOOST = 0.0001f;

    /**
     * Run a BM25 search. When {@code padToK} is true, fill the result up to {@code k} hits
     * with low-scored {@link MatchAllDocsQuery} matches so the response always has the same
     * cardinality as $vectorSearch (handy for side-by-side UI comparison). Each hit carries
     * a {@code matched} flag so callers can distinguish real BM25 matches from filler.
     */
    public synchronized List<Map<String, Object>> search(String q, int k, boolean padToK) throws Exception {
        if (q == null || q.isBlank()) {
            return List.of();
        }
        var parser = new org.apache.lucene.queryparser.classic.QueryParser("text", new StandardAnalyzer());
        Query userQuery = parser.parse(q);

        Query effective;
        if (padToK) {
            BooleanQuery.Builder bq = new BooleanQuery.Builder();
            bq.add(userQuery, BooleanClause.Occur.SHOULD);
            bq.add(new BoostQuery(new MatchAllDocsQuery(), FILL_BOOST), BooleanClause.Occur.SHOULD);
            effective = bq.build();
        } else {
            effective = userQuery;
        }

        IndexSearcher searcher = acquireSearcher();
        TopDocs top = searcher.search(effective, k);
        List<Map<String, Object>> out = new ArrayList<>();
        for (ScoreDoc sd : top.scoreDocs) {
            Document d = searcher.storedFields().document(sd.doc);
            Map<String, Object> row = new HashMap<>();
            row.put("_id", d.get("_id"));
            row.put("score", sd.score);
            row.put("matched", sd.score > MATCH_THRESHOLD);
            out.add(row);
        }
        return out;
    }

    /**
     * Run a kNN search over the dense {@value #EMBEDDING_FIELD} field. This is Lucene's native
     * HNSW vector search and is the same primitive Atlas $vectorSearch is built on top of - the
     * point of the demo is to show the two engines produce equivalent rankings.
     *
     * @param queryVec query embedding (any dimensionality, must match the indexed docs)
     * @param k        top-k to return
     */
    public synchronized List<Map<String, Object>> vectorSearch(float[] queryVec, int k) throws IOException {
        if (queryVec == null || queryVec.length == 0) {
            return List.of();
        }
        KnnFloatVectorQuery q = new KnnFloatVectorQuery(EMBEDDING_FIELD, queryVec, k);
        IndexSearcher searcher = acquireSearcher();
        TopDocs top = searcher.search(q, k);
        int maxDoc = searcher.getIndexReader().maxDoc();
        List<Map<String, Object>> out = new ArrayList<>();
        for (ScoreDoc sd : top.scoreDocs) {
            // KnnFloatVectorQuery may return NO_MORE_DOCS sentinels (Integer.MAX_VALUE) in the
            // tail when HNSW finds fewer than k candidates. Skip them.
            if (sd.doc < 0 || sd.doc >= maxDoc) {
                continue;
            }
            Document d = searcher.storedFields().document(sd.doc);
            Map<String, Object> row = new HashMap<>();
            row.put("_id", d.get("_id"));
            row.put("score", sd.score);
            out.add(row);
        }
        return out;
    }

    /**
     * Score-level hybrid: one Lucene {@link BooleanQuery} that combines a parsed BM25 query and a
     * {@link KnnFloatVectorQuery} as two {@link BooleanClause.Occur#SHOULD} clauses, each wrapped
     * in a {@link BoostQuery} with caller-supplied weights. Lucene's BooleanQuery scorer sums the
     * per-clause sub-scores, so the final score is literally
     * {@code bm25Weight*bm25Score + vecWeight*cosineScore} per document — the score-level hybrid
     * that Solr's {@code function_score}/{@code bf}/{@code bq} and OpenSearch's {@code hybrid}
     * query expose, and that Atlas {@code $rankFusion} (rank-level RRF) does <em>not</em>.
     *
     * <p>For each hit we re-{@link IndexSearcher#explain(Query, int) explain} the score to recover
     * the BM25 and vector contributions separately, so the UI can show the breakdown.
     */
    public synchronized List<Map<String, Object>> hybridSearch(
            String q, float[] queryVec, float bm25Weight, float vecWeight, int k) throws Exception {
        if ((q == null || q.isBlank()) && (queryVec == null || queryVec.length == 0)) {
            return List.of();
        }
        BooleanQuery.Builder bq = new BooleanQuery.Builder();
        Query bm25Boosted = null;
        Query knnBoosted = null;
        if (q != null && !q.isBlank() && bm25Weight > 0f) {
            var parser = new org.apache.lucene.queryparser.classic.QueryParser("text", new StandardAnalyzer());
            bm25Boosted = new BoostQuery(parser.parse(q), bm25Weight);
            bq.add(bm25Boosted, BooleanClause.Occur.SHOULD);
        }
        if (queryVec != null && queryVec.length > 0 && vecWeight > 0f) {
            knnBoosted = new BoostQuery(
                    new KnnFloatVectorQuery(EMBEDDING_FIELD, queryVec, Math.max(k, 10)), vecWeight);
            bq.add(knnBoosted, BooleanClause.Occur.SHOULD);
        }
        BooleanQuery query = bq.build();

        IndexSearcher searcher = acquireSearcher();
        TopDocs top = searcher.search(query, k);
        int maxDoc = searcher.getIndexReader().maxDoc();
        List<Map<String, Object>> out = new ArrayList<>();
        for (ScoreDoc sd : top.scoreDocs) {
            if (sd.doc < 0 || sd.doc >= maxDoc) {
                continue;
            }
            Document d = searcher.storedFields().document(sd.doc);
            // Per-clause contributions: explain each SHOULD clause SEPARATELY against the doc.
            // BooleanQuery's combined explanation only includes details for matched clauses, so
            // its detail order can't be trusted as "[bm25, vec]" — for vec-only matches you'd
            // get [vec] and mistake it for bm25. Explaining each sub-query gives unambiguous
            // contributions even when one clause didn't match.
            float bm25Contrib = 0f;
            float vecContrib = 0f;
            if (bm25Boosted != null) {
                Explanation ex = searcher.explain(bm25Boosted, sd.doc);
                if (ex.isMatch()) bm25Contrib = ex.getValue().floatValue();
            }
            if (knnBoosted != null) {
                Explanation ex = searcher.explain(knnBoosted, sd.doc);
                if (ex.isMatch()) vecContrib = ex.getValue().floatValue();
            }
            Map<String, Object> row = new HashMap<>();
            row.put("_id", d.get("_id"));
            row.put("score", sd.score);
            row.put("bm25_contrib", bm25Contrib);
            row.put("vec_contrib", vecContrib);
            row.put("bm25_matched", bm25Contrib > MATCH_THRESHOLD);
            out.add(row);
        }
        return out;
    }

    /** Lenient JSON-array → float[] coercion. Gson hands us {@code List<Double>} for numeric arrays. */
    private static float[] toFloatArray(Object o) {
        if (o == null) {
            return null;
        }
        if (o instanceof float[] f) {
            return f;
        }
        if (o instanceof double[] d) {
            float[] out = new float[d.length];
            for (int i = 0; i < d.length; i++) {
                out[i] = (float) d[i];
            }
            return out;
        }
        if (o instanceof List<?> list) {
            float[] out = new float[list.size()];
            for (int i = 0; i < list.size(); i++) {
                Object v = list.get(i);
                if (v instanceof Number n) {
                    out[i] = n.floatValue();
                } else {
                    return null;
                }
            }
            return out;
        }
        return null;
    }

    @Override
    public synchronized void close() throws IOException {
        if (reader != null) {
            reader.close();
            reader = null;
        }
        writer.close();
        directory.close();
    }
}
