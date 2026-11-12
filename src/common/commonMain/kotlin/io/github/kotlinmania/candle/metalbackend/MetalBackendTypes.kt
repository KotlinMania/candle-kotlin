package io.github.kotlinmania.candle.metalbackend

import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candlemetalkernels.MetalKernelError
import io.github.kotlinmania.rustlang.std.sync.PoisonError
import io.github.kotlinmania.rustlang.std.sync.TryLockError

/**
 * Simple way to catch lock error without
 * depending on T
 */
public sealed class LockError {
    public data class Poisoned(
        public var value: String,
    ) : LockError()
    public data object WouldBlock : LockError()
}

/**
 * Metal related errors
 */
public sealed class MetalError : From<TryLockError<T>> {
    public data class Message(
        public var value: String,
    ) : MetalError()
    public data class KernelError(
        public var value: MetalKernelError,
    ) : MetalError()
    public data class LockError(
        public var value: LockError,
    ) : MetalError()
    public data class UnexpectedDType(
        public var msg: String,
        public var expected: DType,
        public var got: DType,
    ) : MetalError()

    public companion object {
        public fun <T> from(value: TryLockError<T>): MetalError =
            when (value) {
                is TryLockError.Poisoned -> {
                    val (p) = value
                    MetalError.LockError(LockError.Poisoned(p.toString()))
                }
                TryLockError.WouldBlock -> MetalError.LockError(LockError.WouldBlock)
            }

        public fun <T> from(p: PoisonError<T>): MetalError = MetalError.LockError(LockError.Poisoned(p.toString()))

        public fun from(e: String): MetalError = MetalError.Message(e)
    }
}
