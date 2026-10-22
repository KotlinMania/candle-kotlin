import java.util.Properties
import org.gradle.process.ExecOperations
import org.gradle.kotlin.dsl.support.serviceOf
import org.gradle.api.artifacts.component.ModuleComponentIdentifier
import org.gradle.api.artifacts.component.ModuleComponentSelector
import org.gradle.api.artifacts.result.ResolvedDependencyResult

plugins {
    alias(libs.plugins.kotlin.multiplatform)
    alias(libs.plugins.android.kmp)
}
val producer = Properties().apply { file("../../gradle.properties").inputStream().use { load(it) } }
val libraryVersion = producer.getProperty("project.version")
val hostPlatform = when {
    System.getProperty("os.name").lowercase().contains("mac") -> "macos"
    System.getProperty("os.name").lowercase().contains("windows") -> "windows"
    else -> "linux"
}
val hostArchitecture = if (System.getProperty("os.arch").lowercase() in listOf("arm64", "aarch64")) "aarch64" else "x86_64"
val extractHostJni = tasks.register<Sync>("extractPublishedHostJni") {
    from(zipTree(file("../../build/libs/candle-kotlin-jvm-$libraryVersion.jar"))) {
        include("io/github/kotlinmania/candle/native/$hostPlatform-$hostArchitecture/*")
        eachFile { path = name }
        includeEmptyDirs = false
    }
    into(layout.buildDirectory.dir("published-host-jni"))
}
kotlin {
    jvmToolchain(producer.getProperty("jvm.toolchain").toInt())
    compilerOptions { allWarningsAsErrors.set(true) }
    android {
        namespace = "io.github.kotlinmania.candle.artifactconsumer"
        compileSdk = producer.getProperty("android.compileSdk").toInt()
        minSdk = producer.getProperty("android.minSdk").toInt()
        withHostTestBuilder { sourceSetTreeName = "test" }
    }
    sourceSets {
        commonMain.dependencies { implementation("io.github.kotlinmania:candle-kotlin:$libraryVersion") }
        commonTest.dependencies { implementation(kotlin("test")) }
    }
}
val consumerTests = tasks.withType<Test>().matching { it.name == "testAndroidHostTest" }
consumerTests.configureEach {
    dependsOn(extractHostJni)
    outputs.upToDateWhen { false }
    outputs.doNotCacheIf("Consumer tests must execute") { true }
    systemProperty("java.library.path", layout.buildDirectory.dir("published-host-jni").get().asFile.absolutePath)
    doFirst { layout.buildDirectory.dir("consumer-io").get().asFile.mkdirs() }
    testLogging { events("passed", "failed", "skipped") }
}
kotlin.targets.named("android") {
    compilations.named("main") {
        val dependencies = configurations.getByName(compileDependencyConfigurationName)
        val selection = dependencies.incoming.resolutionResult.rootComponent.map { root ->
            val dependency = root.dependencies.filterIsInstance<ResolvedDependencyResult>().single {
                val requested = it.requested
                requested is ModuleComponentSelector && requested.group == "io.github.kotlinmania" && requested.module == "candle-kotlin"
            }
            val variant = dependency.resolvedVariant.externalVariant.orElse(dependency.resolvedVariant)
            val owner = variant.owner as ModuleComponentIdentifier
            variant.attributes.keySet().associate { it.name to variant.attributes.getAttribute(it).toString() } +
                mapOf("group" to owner.group, "module" to owner.module, "version" to owner.version)
        }
        tasks.register("verifyAndroidPublication") {
            dependsOn(consumerTests)
            doLast {
                val actual = selection.get()
                mapOf("group" to "io.github.kotlinmania", "module" to "candle-kotlin-android", "version" to libraryVersion,
                    "org.jetbrains.kotlin.platform.type" to "androidJvm", "org.gradle.libraryelements" to "aar",
                    "org.gradle.jvm.environment" to "android").forEach { (key, value) ->
                    check(actual[key] == value) { "Wrong artifact selection: $actual; expected $key=$value" }
                }
                check(consumerTests.size == 1)
                project.serviceOf<ExecOperations>().exec {
                    commandLine("python3", "../verify_behavior_results.py", consumerTests.single().reports.junitXml.outputLocation.get().asFile,
                        "--require", "publishedAndroidApiCreatesTensor")
                }.assertNormalExitValue()
                layout.buildDirectory.file("reports/selected-android-artifact.json").get().asFile.apply {
                    parentFile.mkdirs()
                    writeText(groovy.json.JsonOutput.toJson(actual) + "\n")
                }
            }
        }
    }
}
