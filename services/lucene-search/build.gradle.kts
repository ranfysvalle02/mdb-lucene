plugins {
    java
    application
}

group = "io.homecook"
version = "1.0.0"

java {
    toolchain {
        languageVersion.set(JavaLanguageVersion.of(21))
    }
}

repositories {
    mavenCentral()
}

dependencies {
    implementation("io.javalin:javalin:5.6.3")
    implementation("org.apache.lucene:lucene-core:9.11.1")
    implementation("org.apache.lucene:lucene-queryparser:9.11.1")
    implementation("org.apache.lucene:lucene-analysis-common:9.11.1")
    implementation("com.google.code.gson:gson:2.11.0")
    implementation("org.slf4j:slf4j-simple:2.0.16")
}

application {
    mainClass.set("io.homecook.lucene.App")
    applicationName = "lucene-search"
}

tasks.withType<JavaCompile>().configureEach {
    options.encoding = "UTF-8"
}
