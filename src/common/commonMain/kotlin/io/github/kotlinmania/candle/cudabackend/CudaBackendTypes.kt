package io.github.kotlinmania.candle.cudabackend

import io.github.kotlinmania.candle.cudabackend.device.CudaDevice
import io.github.kotlinmania.candle.cudabackend.device.DeviceId
import io.github.kotlinmania.candle.cudabackend.device.cudaGraphHtodCacheEnabled
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.cudarc.driver.CudaSlice
import io.github.kotlinmania.cudarc.driver.LaunchArgs
import io.github.kotlinmania.float8.F8e4m3
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.rustlang.core.result.fromResidual
import io.github.kotlinmania.rustlang.std.sync.Mutex

public typealias ParamCache = Map<Pair<DeviceId, List<Int>>, CudaSlice<Int>>

public sealed class SlicePtrOrNull<T> {
    public data class Ptr<T>(
        public var value: CudaSlice<T>,
    ) : SlicePtrOrNull<T>()
    public data class Cached<T>(
        public var value: CudaSlice<T>,
    ) : SlicePtrOrNull<T>()
    public class Null<T> : SlicePtrOrNull<T>()

    public fun builderArg(builder: LaunchArgs) {
        when (this) {
            is SlicePtrOrNull.Ptr -> {
                val (slice) = this
                builder.arg(slice)
            }
            is SlicePtrOrNull.Cached -> {
                val (slice1) = this
                builder.arg(slice1.asRef())
            }
            SlicePtrOrNull.Null -> builder.arg(0)
        }
    }
}

public fun SlicePtrOrNull.Companion.paramsFromVec(dev: CudaDevice, params: List<Int>): Result<SlicePtrOrNull<Int>> {
    if (cudaGraphHtodCacheEnabled()) {
        val key = Pair(dev.id(), params.clone())
        val cache = CUDA_PARAM_CACHE.getOrInit({ Mutex.new(HashMap.new()) })
        when (
            val matchVal =
                cache
                    .lock()
                    .unwrap()
                    .get(key)
                    .cloned()
        ) {
            null -> {}
            else -> {
                val slice = matchVal
                return Ok(SlicePtrOrNull.Cached(slice))
            }
        }
        val isCapturing = (dev.cudaStream().captureStatus()).rustEquals(Ok(CUstreamCaptureStatus.CU_STREAM_CAPTURE_STATUS_ACTIVE))
        if (isCapturing) {
            bail("h2d param cache miss during CUDA graph capture: warmup did not populate all required parameter vectors")
        }
        val slice1 =
            when (val tryResult = dev.cudaStream().cloneHtod(params).w()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val cache1 = cache.lock().unwrap()
        val slice2 = cache1.entry(key).orInsertWith({ slice1 }).clone()
        return Ok(SlicePtrOrNull.Cached(slice2))
    }
    return Ok(
        SlicePtrOrNull.Ptr(
            when (val tryResult = dev.cloneHtod(params)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            },
        ),
    )
}

public fun SlicePtrOrNull.Companion.paramsFromLayout(dev: CudaDevice, l: Layout): Result<SlicePtrOrNull<Int>> =
    if (l.isContiguous()) {
        Ok(SlicePtrOrNull.Null)
    } else {
        SlicePtrOrNull.paramsFromVec(dev, listOf(l.dims(), l.stride()).concat())
    }

public sealed class CudaStorageSlice {
    public class U8 internal constructor(
        valueCtorArg: CudaSlice<UByte>,
    ) : CudaStorageSlice() {
        public var value: CudaSlice<UByte> = valueCtorArg
            internal set

        public operator fun component1(): CudaSlice<UByte> = this.value

        public fun copy(value: CudaSlice<UByte> = this.value): U8 = U8(valueCtorArg = value)

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is U8) return false
            if (this.value != other.value) return false
            return true
        }

        override fun hashCode(): Int = this.value.hashCode()

        public companion object {
            public operator fun invoke(
                value: CudaSlice<UByte>,
            ): U8 =
                U8(
                    valueCtorArg = value,
                )
        }
    }
    public class U32 internal constructor(
        valueCtorArg: CudaSlice<UInt>,
    ) : CudaStorageSlice() {
        public var value: CudaSlice<UInt> = valueCtorArg
            internal set

        public operator fun component1(): CudaSlice<UInt> = this.value

        public fun copy(value: CudaSlice<UInt> = this.value): U32 = U32(valueCtorArg = value)

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is U32) return false
            if (this.value != other.value) return false
            return true
        }

        override fun hashCode(): Int = this.value.hashCode()

        public companion object {
            public operator fun invoke(
                value: CudaSlice<UInt>,
            ): U32 =
                U32(
                    valueCtorArg = value,
                )
        }
    }
    public class I16 internal constructor(
        valueCtorArg: CudaSlice<Short>,
    ) : CudaStorageSlice() {
        public var value: CudaSlice<Short> = valueCtorArg
            internal set

        public operator fun component1(): CudaSlice<Short> = this.value

        public fun copy(value: CudaSlice<Short> = this.value): I16 = I16(valueCtorArg = value)

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is I16) return false
            if (this.value != other.value) return false
            return true
        }

        override fun hashCode(): Int = this.value.hashCode()

        public companion object {
            public operator fun invoke(
                value: CudaSlice<Short>,
            ): I16 =
                I16(
                    valueCtorArg = value,
                )
        }
    }
    public class I32 internal constructor(
        valueCtorArg: CudaSlice<Int>,
    ) : CudaStorageSlice() {
        public var value: CudaSlice<Int> = valueCtorArg
            internal set

        public operator fun component1(): CudaSlice<Int> = this.value

        public fun copy(value: CudaSlice<Int> = this.value): I32 = I32(valueCtorArg = value)

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is I32) return false
            if (this.value != other.value) return false
            return true
        }

        override fun hashCode(): Int = this.value.hashCode()

        public companion object {
            public operator fun invoke(
                value: CudaSlice<Int>,
            ): I32 =
                I32(
                    valueCtorArg = value,
                )
        }
    }
    public class I64 internal constructor(
        valueCtorArg: CudaSlice<Long>,
    ) : CudaStorageSlice() {
        public var value: CudaSlice<Long> = valueCtorArg
            internal set

        public operator fun component1(): CudaSlice<Long> = this.value

        public fun copy(value: CudaSlice<Long> = this.value): I64 = I64(valueCtorArg = value)

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is I64) return false
            if (this.value != other.value) return false
            return true
        }

        override fun hashCode(): Int = this.value.hashCode()

        public companion object {
            public operator fun invoke(
                value: CudaSlice<Long>,
            ): I64 =
                I64(
                    valueCtorArg = value,
                )
        }
    }
    public class Bf16 internal constructor(
        valueCtorArg: CudaSlice<Bf16>,
    ) : CudaStorageSlice() {
        public var value: CudaSlice<Bf16> = valueCtorArg
            internal set

        public operator fun component1(): CudaSlice<Bf16> = this.value

        public fun copy(value: CudaSlice<Bf16> = this.value): Bf16 = Bf16(valueCtorArg = value)

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is Bf16) return false
            if (this.value != other.value) return false
            return true
        }

        override fun hashCode(): Int = this.value.hashCode()

        public companion object {
            public operator fun invoke(
                value: CudaSlice<Bf16>,
            ): Bf16 =
                Bf16(
                    valueCtorArg = value,
                )
        }
    }
    public class F16 internal constructor(
        valueCtorArg: CudaSlice<F16>,
    ) : CudaStorageSlice() {
        public var value: CudaSlice<F16> = valueCtorArg
            internal set

        public operator fun component1(): CudaSlice<F16> = this.value

        public fun copy(value: CudaSlice<F16> = this.value): F16 = F16(valueCtorArg = value)

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is F16) return false
            if (this.value != other.value) return false
            return true
        }

        override fun hashCode(): Int = this.value.hashCode()

        public companion object {
            public operator fun invoke(
                value: CudaSlice<F16>,
            ): F16 =
                F16(
                    valueCtorArg = value,
                )
        }
    }
    public class F32 internal constructor(
        valueCtorArg: CudaSlice<Float>,
    ) : CudaStorageSlice() {
        public var value: CudaSlice<Float> = valueCtorArg
            internal set

        public operator fun component1(): CudaSlice<Float> = this.value

        public fun copy(value: CudaSlice<Float> = this.value): F32 = F32(valueCtorArg = value)

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is F32) return false
            if (this.value != other.value) return false
            return true
        }

        override fun hashCode(): Int = this.value.hashCode()

        public companion object {
            public operator fun invoke(
                value: CudaSlice<Float>,
            ): F32 =
                F32(
                    valueCtorArg = value,
                )
        }
    }
    public class F64 internal constructor(
        valueCtorArg: CudaSlice<Double>,
    ) : CudaStorageSlice() {
        public var value: CudaSlice<Double> = valueCtorArg
            internal set

        public operator fun component1(): CudaSlice<Double> = this.value

        public fun copy(value: CudaSlice<Double> = this.value): F64 = F64(valueCtorArg = value)

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is F64) return false
            if (this.value != other.value) return false
            return true
        }

        override fun hashCode(): Int = this.value.hashCode()

        public companion object {
            public operator fun invoke(
                value: CudaSlice<Double>,
            ): F64 =
                F64(
                    valueCtorArg = value,
                )
        }
    }
    public class F8e4m3 internal constructor(
        valueCtorArg: CudaSlice<F8e4m3>,
    ) : CudaStorageSlice() {
        public var value: CudaSlice<F8e4m3> = valueCtorArg
            internal set

        public operator fun component1(): CudaSlice<F8e4m3> = this.value

        public fun copy(value: CudaSlice<F8e4m3> = this.value): F8e4m3 = F8e4m3(valueCtorArg = value)

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is F8e4m3) return false
            if (this.value != other.value) return false
            return true
        }

        override fun hashCode(): Int = this.value.hashCode()

        public companion object {
            public operator fun invoke(
                value: CudaSlice<F8e4m3>,
            ): F8e4m3 =
                F8e4m3(
                    valueCtorArg = value,
                )
        }
    }

    // Dummy types that store raw bytes
    public class F6e2m3 internal constructor(
        valueCtorArg: CudaSlice<UByte>,
    ) : CudaStorageSlice() {
        public var value: CudaSlice<UByte> = valueCtorArg
            internal set

        public operator fun component1(): CudaSlice<UByte> = this.value

        public fun copy(value: CudaSlice<UByte> = this.value): F6e2m3 = F6e2m3(valueCtorArg = value)

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is F6e2m3) return false
            if (this.value != other.value) return false
            return true
        }

        override fun hashCode(): Int = this.value.hashCode()

        public companion object {
            public operator fun invoke(
                value: CudaSlice<UByte>,
            ): F6e2m3 =
                F6e2m3(
                    valueCtorArg = value,
                )
        }
    }
    public class F6e3m2 internal constructor(
        valueCtorArg: CudaSlice<UByte>,
    ) : CudaStorageSlice() {
        public var value: CudaSlice<UByte> = valueCtorArg
            internal set

        public operator fun component1(): CudaSlice<UByte> = this.value

        public fun copy(value: CudaSlice<UByte> = this.value): F6e3m2 = F6e3m2(valueCtorArg = value)

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is F6e3m2) return false
            if (this.value != other.value) return false
            return true
        }

        override fun hashCode(): Int = this.value.hashCode()

        public companion object {
            public operator fun invoke(
                value: CudaSlice<UByte>,
            ): F6e3m2 =
                F6e3m2(
                    valueCtorArg = value,
                )
        }
    }
    public class F4 internal constructor(
        valueCtorArg: CudaSlice<UByte>,
    ) : CudaStorageSlice() {
        public var value: CudaSlice<UByte> = valueCtorArg
            internal set

        public operator fun component1(): CudaSlice<UByte> = this.value

        public fun copy(value: CudaSlice<UByte> = this.value): F4 = F4(valueCtorArg = value)

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is F4) return false
            if (this.value != other.value) return false
            return true
        }

        override fun hashCode(): Int = this.value.hashCode()

        public companion object {
            public operator fun invoke(
                value: CudaSlice<UByte>,
            ): F4 =
                F4(
                    valueCtorArg = value,
                )
        }
    }
    public class F8e8m0 internal constructor(
        valueCtorArg: CudaSlice<UByte>,
    ) : CudaStorageSlice() {
        public var value: CudaSlice<UByte> = valueCtorArg
            internal set

        public operator fun component1(): CudaSlice<UByte> = this.value

        public fun copy(value: CudaSlice<UByte> = this.value): F8e8m0 = F8e8m0(valueCtorArg = value)

        override fun equals(other: Any?): Boolean {
            if (this === other) return true
            if (other !is F8e8m0) return false
            if (this.value != other.value) return false
            return true
        }

        override fun hashCode(): Int = this.value.hashCode()

        public companion object {
            public operator fun invoke(
                value: CudaSlice<UByte>,
            ): F8e8m0 =
                F8e8m0(
                    valueCtorArg = value,
                )
        }
    }
}

public enum class PoolOp(
    public val value: UInt,
) {
    Max(0u),
    Avg(1u),
}

public interface CudaDType<T> {
    public fun asCudaSlice(s: CudaStorage): Result<CudaSlice<T>>

    public fun asCudaSliceMut(s: CudaStorage): Result<CudaSlice<T>>

    public fun wrapCudaSlice(s: CudaSlice<T>, dev: CudaDevice): CudaStorage
}
