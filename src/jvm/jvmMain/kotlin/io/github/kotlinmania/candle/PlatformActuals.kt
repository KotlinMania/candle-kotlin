// Platform-specific feature detection for JVM target.
// Replaces stubbed cfg/feature values from the Rust-to-Kotlin port.
package io.github.kotlinmania.candle

internal actual val targetOs: String =
    System.getProperty("os.name").lowercase().let {
        when {
            it.contains("mac") -> "macos"
            it.contains("linux") -> "linux"
            it.contains("windows") -> "windows"
            else -> it
        }
    }

internal actual val feature: String = if (targetOs == "macos") "accelerate" else ""

internal actual val targetFeature: String = ""