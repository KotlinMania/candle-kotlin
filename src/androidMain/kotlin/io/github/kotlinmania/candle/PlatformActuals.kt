// Platform-specific feature detection for Android target.
// Android does not have Accelerate; ARM64 has NEON.
package io.github.kotlinmania.candle

internal actual val targetOs: String = "android"

internal actual val feature: String = ""

internal actual val targetFeature: String = "neon"