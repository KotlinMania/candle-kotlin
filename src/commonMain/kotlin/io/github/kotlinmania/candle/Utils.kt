// port-lint: source candle-core/src/utils.rs
package io.github.kotlinmania.candle

public fun getNumThreads(): Int = defaultNumThreads()

public fun defaultNumThreads(): Int = 4

public fun <R> withThreadpool(f: () -> R): R = f()

public fun hasAccelerate(): Boolean = false

public fun hasMkl(): Boolean = false

public fun cudaIsAvailable(): Boolean = false

public fun metalIsAvailable(): Boolean = false

public fun withAvx(): Boolean = false

public fun withNeon(): Boolean = false

public fun withSimd128(): Boolean = false

public fun withF16c(): Boolean = false
