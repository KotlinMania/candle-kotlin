pluginManagement {
    repositories { google(); mavenCentral(); gradlePluginPortal() }
}
dependencyResolutionManagement {
    repositories {
        exclusiveContent {
            forRepository { maven { url = uri("../../build/staging-deploy") } }
            filter {
                includeModule("io.github.kotlinmania", "candle-kotlin")
                includeModule("io.github.kotlinmania", "candle-kotlin-android")
            }
        }
        google()
        mavenCentral()
    }
    versionCatalogs { create("libs") { from(files("../../gradle/libs.versions.toml")) } }
}
rootProject.name = "candle-android-artifact-consumer"
