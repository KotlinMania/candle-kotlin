// Platform-specific feature detection for Kotlin/Native targets.
// Uses Platform.osFamily / Platform.cpuArchitecture for compile-time
// target detection, replacing the stubbed values from the Rust port.
package io.github.kotlinmania.candle

import kotlin.experimental.ExperimentalNativeApi
import kotlin.native.Platform

@OptIn(ExperimentalNativeApi::class)
internal actual val targetOs: String = Platform.osFamily.name.lowercase()

internal actual val feature: String = if (targetOs == "macos") "accelerate" else ""

@OptIn(ExperimentalNativeApi::class)
internal actual val targetFeature: String =
    when (Platform.cpuArchitecture.name.lowercase()) {
        "arm64" -> "neon"
        "x64" -> "avx2"
        else -> ""
    }