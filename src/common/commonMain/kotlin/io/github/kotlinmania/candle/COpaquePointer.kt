package io.github.kotlinmania.candle

import kotlin.jvm.JvmInline

// Pure Kotlin opaque pointer — platform-neutral pointer representation.
// On Native, cinterop provides kotlinx.cinterop.COpaquePointer; this class
// bridges the common API so expect/actual declarations work on all targets.
@JvmInline
public value class COpaquePointer(
    public val value: Long = 0L,
) {
    public val address: Long get() = value
    public val rawValue: Long get() = value
}

public fun COpaquePointer?.add(count: Int): COpaquePointer? =
    this?.let { COpaquePointer(it.value + count.toLong()) }

public fun COpaquePointer?.add(count: Long): COpaquePointer? =
    this?.let { COpaquePointer(it.value + count) }

public fun COpaquePointer?.add(count: UInt): COpaquePointer? =
    this?.let { COpaquePointer(it.value + count.toLong()) }

public fun COpaquePointer?.add(count: ULong): COpaquePointer? =
    this?.let { COpaquePointer(it.value + count.toLong()) }

public fun COpaquePointer?.offset(count: Int): COpaquePointer? =
    this?.let { COpaquePointer(it.value + count.toLong()) }

public fun COpaquePointer?.offset(count: Long): COpaquePointer? =
    this?.let { COpaquePointer(it.value + count) }

public fun COpaquePointer?.cast(): COpaquePointer? = this

public operator fun COpaquePointer?.compareTo(other: COpaquePointer?): Int =
    (this?.value ?: 0L).toULong().compareTo((other?.value ?: 0L).toULong())
