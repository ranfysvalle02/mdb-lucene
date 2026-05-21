FROM gradle:8.10.2-jdk21 AS build
WORKDIR /app
COPY settings.gradle.kts build.gradle.kts ./
COPY src ./src
RUN gradle --no-daemon installDist

FROM eclipse-temurin:21-jre-jammy

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/lucene-search

COPY --from=build /app/build/install/lucene-search /opt/lucene-search

ENV LUCENE_INDEX_DIR=/app/lucene/index
ENV LUCENE_HTTP_PORT=9090

RUN mkdir -p /app/lucene/index \
    && chmod +x /opt/lucene-search/bin/lucene-search

EXPOSE 9090

ENTRYPOINT ["/opt/lucene-search/bin/lucene-search"]
