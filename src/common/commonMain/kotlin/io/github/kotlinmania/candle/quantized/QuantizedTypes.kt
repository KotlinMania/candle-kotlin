package io.github.kotlinmania.candle.quantized

import io.github.kotlinmania.candle.Module
import io.github.kotlinmania.candle.cpubackend.CpuStorage
import io.github.kotlinmania.candle.device.Device
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.quantized.kquants.BlockQ2K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ3K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ40
import io.github.kotlinmania.candle.quantized.kquants.BlockQ41
import io.github.kotlinmania.candle.quantized.kquants.BlockQ4K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ50
import io.github.kotlinmania.candle.quantized.kquants.BlockQ51
import io.github.kotlinmania.candle.quantized.kquants.BlockQ5K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ6K
import io.github.kotlinmania.candle.quantized.kquants.BlockQ80
import io.github.kotlinmania.candle.quantized.kquants.BlockQ81
import io.github.kotlinmania.candle.quantized.kquants.BlockQ8K
import io.github.kotlinmania.candle.quantized.kquants.QK4_0
import io.github.kotlinmania.candle.quantized.kquants.QK4_1
import io.github.kotlinmania.candle.quantized.kquants.QK5_0
import io.github.kotlinmania.candle.quantized.kquants.QK5_1
import io.github.kotlinmania.candle.quantized.kquants.QK8_0
import io.github.kotlinmania.candle.quantized.kquants.QK8_1
import io.github.kotlinmania.candle.quantized.kquants.QK_K
import io.github.kotlinmania.candle.shape.D
import io.github.kotlinmania.candle.storage.Storage
import io.github.kotlinmania.candle.tensor.Tensor
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.rustlang.core.mem.ObjectSlotLayout
import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.result.fromResidual
import io.github.kotlinmania.rustlang.std.borrow.Cow

public sealed class QStorage {
    public data class Cpu(
        public var value: QuantizedType,
    ) : QStorage()
    public data class Metal(
        public var value: QMetalStorage,
    ) : QStorage()
    public data class Cuda(
        public var value: QCudaStorage,
    ) : QStorage()

    public fun blockSize(): Int =
        when (this) {
            is QStorage.Cpu -> {
                val (storage) = this
                storage.blockSize()
            }
            is QStorage.Metal -> {
                val (storage1) = this
                storage1.dtype().blockSize()
            }
            is QStorage.Cuda -> {
                val (storage2) = this
                storage2.dtype().blockSize()
            }
        }

    public fun dtype(): GgmlDType =
        when (this) {
            is QStorage.Cpu -> {
                val (storage) = this
                storage.dtype()
            }
            is QStorage.Metal -> {
                val (storage1) = this
                storage1.dtype()
            }
            is QStorage.Cuda -> {
                val (storage2) = this
                storage2.dtype()
            }
        }

    public fun device(): Device =
        when (this) {
            is QStorage.Cpu -> Device.Cpu
            is QStorage.Metal -> {
                val (storage) = this
                Device.Metal(storage.device().clone())
            }
            is QStorage.Cuda -> {
                val (storage1) = this
                Device.Cuda(storage1.device().clone())
            }
        }

    public fun sizeInBytes(): Int =
        when (this) {
            is QStorage.Cpu -> {
                val (storage) = this
                storage.storageSizeInBytes()
            }
            is QStorage.Metal -> {
                val (storage1) = this
                storage1.storageSizeInBytes()
            }
            is QStorage.Cuda -> {
                val (storage2) = this
                storage2.storageSizeInBytes()
            }
        }

    public fun quantize(src: Storage): Result<Unit> {
        run {
            val matchVal = Pair(this, src)
            when {
                matchVal.first is QStorage.Cpu && matchVal.second is Storage.Cpu -> {
                    val (storage) = matchVal.first as QStorage.Cpu
                    val (src1) = matchVal.second as Storage.Cpu
                    storage.fromFloat(
                        when (val tryResult = src1.asSlice<Float>()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                }
                matchVal.first is QStorage.Metal && matchVal.second is Storage.Metal -> {
                    val (storage1) = matchVal.first as QStorage.Metal
                    val (src2) = matchVal.second as Storage.Metal
                    when (val tryResult = storage1.quantize(src2)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
                matchVal.first is QStorage.Cuda && matchVal.second is Storage.Cuda -> {
                    val (storage2) = matchVal.first as QStorage.Cuda
                    val (src3) = matchVal.second as Storage.Cuda
                    when (val tryResult = storage2.quantize(src3)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
                else -> bail("Invalid quantize storage locations do not match")
            }
        }
        return Ok(Unit)
    }

    public fun quantizeImatrix(src: Storage, imatrixWeights: List<Float>, nPerRow: Int): Result<Unit> {
        run {
            val matchVal = Pair(this, src)
            when {
                matchVal.first is QStorage.Cpu && matchVal.second is Storage.Cpu -> {
                    val (storage) = matchVal.first as QStorage.Cpu
                    val (src1) = matchVal.second as Storage.Cpu
                    storage.fromFloatImatrix(
                        when (val tryResult = src1.asSlice<Float>()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                        imatrixWeights,
                        nPerRow,
                    )
                }
                matchVal.first is QStorage.Metal && matchVal.second is Storage.Metal -> {
                    val (storage1) = matchVal.first as QStorage.Metal
                    val (src2) = matchVal.second as Storage.Metal
                    when (val tryResult = storage1.quantizeImatrix(src2, imatrixWeights, nPerRow)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
                matchVal.first is QStorage.Cuda && matchVal.second is Storage.Cuda -> {
                    val (storage2) = matchVal.first as QStorage.Cuda
                    val (src3) = matchVal.second as Storage.Cuda
                    when (val tryResult = storage2.quantizeImatrix(src3, imatrixWeights, nPerRow)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
                else -> bail("Invalid quantize storage locations do not match")
            }
        }
        return Ok(Unit)
    }

    public fun quantizeOnto(src: Storage): Result<Unit> {
        run {
            val matchVal = Pair(this, src)
            when {
                matchVal.first is QStorage.Cpu && matchVal.second is Storage.Cpu -> {
                    val (storage) = matchVal.first as QStorage.Cpu
                    val (src1) = matchVal.second as Storage.Cpu
                    storage.fromFloat(
                        when (val tryResult = src1.asSlice<Float>()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    )
                }
                matchVal.first is QStorage.Metal && matchVal.second is Storage.Cpu -> {
                    val (storage1) = matchVal.first as QStorage.Metal
                    val (src2) = matchVal.second as Storage.Cpu
                    when (val tryResult = storage1.quantizeOnto(src2)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
                matchVal.first is QStorage.Cuda && matchVal.second is Storage.Cpu -> {
                    val (storage2) = matchVal.first as QStorage.Cuda
                    val (src3) = matchVal.second as Storage.Cpu
                    when (val tryResult = storage2.quantizeOnto(src3)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
                else -> bail("Invalid quantize source storage locations: not on cpu")
            }
        }
        return Ok(Unit)
    }

    public fun quantizeImatrixOnto(src: Storage, imatrixWeights: List<Float>, nPerRow: Int): Result<Unit> {
        run {
            val matchVal = Pair(this, src)
            when {
                matchVal.first is QStorage.Cpu && matchVal.second is Storage.Cpu -> {
                    val (storage) = matchVal.first as QStorage.Cpu
                    val (src1) = matchVal.second as Storage.Cpu
                    storage.fromFloatImatrix(
                        when (val tryResult = src1.asSlice<Float>()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                        imatrixWeights,
                        nPerRow,
                    )
                }
                matchVal.first is QStorage.Metal && matchVal.second is Storage.Cpu -> {
                    val (storage1) = matchVal.first as QStorage.Metal
                    val (src2) = matchVal.second as Storage.Cpu
                    when (val tryResult = storage1.quantizeImatrixOnto(src2, imatrixWeights, nPerRow)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
                matchVal.first is QStorage.Cuda && matchVal.second is Storage.Cpu -> {
                    val (storage2) = matchVal.first as QStorage.Cuda
                    val (src3) = matchVal.second as Storage.Cpu
                    when (val tryResult = storage2.quantizeImatrixOnto(src3, imatrixWeights, nPerRow)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                }
                else -> bail("Invalid quantize storage locations do not match")
            }
        }
        return Ok(Unit)
    }

    public fun dequantize(elemCount: Int): Result<Storage> {
        return when (this) {
            is QStorage.Cpu -> {
                val (storage) = this
                Ok(
                    Storage.Cpu(
                        when (val tryResult = storage.dequantize(elemCount)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    ),
                )
            }
            is QStorage.Metal -> {
                val (storage1) = this
                Ok(
                    Storage.Metal(
                        when (val tryResult = storage1.dequantize(elemCount)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    ),
                )
            }
            is QStorage.Cuda -> {
                val (storage2) = this
                Ok(
                    Storage.Cuda(
                        when (val tryResult = storage2.dequantize(elemCount)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    ),
                )
            }
        }
    }

    public fun data(): Result<Cow<List<UByte>>> {
        return when (this) {
            is QStorage.Cpu -> {
                val (storage) = this
                val dataPtr = storage.asPtr()
                val sizeInBytes = storage.storageSizeInBytes()
                val data = fromRawParts(dataPtr, sizeInBytes)
                Ok(Cow.from(data))
            }
            is QStorage.Cuda -> {
                val (storage1) = this
                Ok(
                    Cow.from(
                        when (val tryResult = storage1.data()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    ),
                )
            }
            is QStorage.Metal -> {
                val (storage2) = this
                Ok(
                    Cow.from(
                        when (val tryResult = storage2.data()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        },
                    ),
                )
            }
        }
    }

    public fun devicePtr(): Result<ConstPtr> =
        when (this) {
            is QStorage.Cuda -> {
                val (storage) = this
                storage.devicePtr()
            }
            is QStorage.Metal -> {
                bail("not implemented")
            }
            is QStorage.Cpu -> {
                bail("not implemented")
            }
        }

    public fun devicePtrWithGuard(stream: CudaStream): Result<Pair<ConstPtr, SyncOnDrop>> =
        when (this) {
            is QStorage.Cuda -> {
                val (storage) = this
                storage.devicePtrWithGuard(stream)
            }
            is QStorage.Metal -> {
                bail("not implemented")
            }
            is QStorage.Cpu -> {
                bail("not implemented")
            }
        }

    public companion object {
        public fun fromData(data: Cow<List<UByte>>, device: Device, dtype: GgmlDType): Result<QStorage> {
            val data1: List<UByte> = data
            return when (device) {
                Device.Cpu -> Ok(QStorage.Cpu(dtype.fromData(Cow.Borrowed(data1))))
                is Device.Metal -> {
                    val (d) = device
                    when (dtype) {
                        GgmlDType.f32 -> loadQuantized(d, asTSlice<Float>(data1, typeLayoutT = PrimitiveLayouts.F32))
                        GgmlDType.f16 -> loadQuantized(d, asTSlice<F16>(data1, typeLayoutT = ObjectSlotLayout<F16>()))
                        GgmlDType.q40 -> loadQuantized(d, asTSlice<BlockQ40>(data1, typeLayoutT = ObjectSlotLayout<BlockQ40>()))
                        GgmlDType.q41 -> loadQuantized(d, asTSlice<BlockQ41>(data1, typeLayoutT = ObjectSlotLayout<BlockQ41>()))
                        GgmlDType.q50 -> loadQuantized(d, asTSlice<BlockQ50>(data1, typeLayoutT = ObjectSlotLayout<BlockQ50>()))
                        GgmlDType.q51 -> loadQuantized(d, asTSlice<BlockQ51>(data1, typeLayoutT = ObjectSlotLayout<BlockQ51>()))
                        GgmlDType.q80 -> loadQuantized(d, asTSlice<BlockQ80>(data1, typeLayoutT = ObjectSlotLayout<BlockQ80>()))
                        GgmlDType.q81 -> loadQuantized(d, asTSlice<BlockQ81>(data1, typeLayoutT = ObjectSlotLayout<BlockQ81>()))
                        GgmlDType.q2k -> loadQuantized(d, asTSlice<BlockQ2K>(data1, typeLayoutT = ObjectSlotLayout<BlockQ2K>()))
                        GgmlDType.q3k -> loadQuantized(d, asTSlice<BlockQ3K>(data1, typeLayoutT = ObjectSlotLayout<BlockQ3K>()))
                        GgmlDType.q4k -> loadQuantized(d, asTSlice<BlockQ4K>(data1, typeLayoutT = ObjectSlotLayout<BlockQ4K>()))
                        GgmlDType.q5k -> loadQuantized(d, asTSlice<BlockQ5K>(data1, typeLayoutT = ObjectSlotLayout<BlockQ5K>()))
                        GgmlDType.q6k -> loadQuantized(d, asTSlice<BlockQ6K>(data1, typeLayoutT = ObjectSlotLayout<BlockQ6K>()))
                        GgmlDType.q8k -> loadQuantized(d, asTSlice<BlockQ8K>(data1, typeLayoutT = ObjectSlotLayout<BlockQ8K>()))
                        GgmlDType.bf16 -> loadQuantized(d, asTSlice<Bf16>(data1, typeLayoutT = ObjectSlotLayout<Bf16>()))
                    }
                }
                is Device.Cuda -> {
                    val (d1) = device
                    when (dtype) {
                        GgmlDType.f32 -> loadQuantized(d1, asTSlice<Float>(data1, typeLayoutT = PrimitiveLayouts.F32))
                        GgmlDType.f16 -> loadQuantized(d1, asTSlice<F16>(data1, typeLayoutT = ObjectSlotLayout<F16>()))
                        GgmlDType.q40 -> loadQuantized(d1, asTSlice<BlockQ40>(data1, typeLayoutT = ObjectSlotLayout<BlockQ40>()))
                        GgmlDType.q41 -> loadQuantized(d1, asTSlice<BlockQ41>(data1, typeLayoutT = ObjectSlotLayout<BlockQ41>()))
                        GgmlDType.q50 -> loadQuantized(d1, asTSlice<BlockQ50>(data1, typeLayoutT = ObjectSlotLayout<BlockQ50>()))
                        GgmlDType.q51 -> loadQuantized(d1, asTSlice<BlockQ51>(data1, typeLayoutT = ObjectSlotLayout<BlockQ51>()))
                        GgmlDType.q80 -> loadQuantized(d1, asTSlice<BlockQ80>(data1, typeLayoutT = ObjectSlotLayout<BlockQ80>()))
                        GgmlDType.q81 -> loadQuantized(d1, asTSlice<BlockQ81>(data1, typeLayoutT = ObjectSlotLayout<BlockQ81>()))
                        GgmlDType.q2k -> loadQuantized(d1, asTSlice<BlockQ2K>(data1, typeLayoutT = ObjectSlotLayout<BlockQ2K>()))
                        GgmlDType.q3k -> loadQuantized(d1, asTSlice<BlockQ3K>(data1, typeLayoutT = ObjectSlotLayout<BlockQ3K>()))
                        GgmlDType.q4k -> loadQuantized(d1, asTSlice<BlockQ4K>(data1, typeLayoutT = ObjectSlotLayout<BlockQ4K>()))
                        GgmlDType.q5k -> loadQuantized(d1, asTSlice<BlockQ5K>(data1, typeLayoutT = ObjectSlotLayout<BlockQ5K>()))
                        GgmlDType.q6k -> loadQuantized(d1, asTSlice<BlockQ6K>(data1, typeLayoutT = ObjectSlotLayout<BlockQ6K>()))
                        GgmlDType.q8k -> loadQuantized(d1, asTSlice<BlockQ8K>(data1, typeLayoutT = ObjectSlotLayout<BlockQ8K>()))
                        GgmlDType.bf16 -> loadQuantized(d1, asTSlice<Bf16>(data1, typeLayoutT = ObjectSlotLayout<Bf16>()))
                    }
                }
            }
        }
    }
}

public enum class GgmlDType(
    public val value: UInt,
) {
    F32(0u),
    F16(1u),
    BF16(2u),
    Q4_0(3u),
    Q4_1(4u),
    Q5_0(5u),
    Q5_1(6u),
    Q8_0(7u),
    Q8_1(8u),
    Q2K(9u),
    Q3K(10u),
    Q4K(11u),
    Q5K(12u),
    Q6K(13u),
    Q8K(14u),
    ;

    internal fun toU32(): UInt =
        when (this) {
            GgmlDType.f32 -> 0u
            GgmlDType.f16 -> 1u
            GgmlDType.q40 -> 2u
            GgmlDType.q41 -> 3u
            GgmlDType.q50 -> 6u
            GgmlDType.q51 -> 7u
            GgmlDType.q80 -> 8u
            GgmlDType.q81 -> 9u
            GgmlDType.q2k -> 10u
            GgmlDType.q3k -> 11u
            GgmlDType.q4k -> 12u
            GgmlDType.q5k -> 13u
            GgmlDType.q6k -> 14u
            GgmlDType.q8k -> 15u
            // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
            GgmlDType.bf16 -> 30u
        }

    /**
     * The block dtype
     */
    public fun cpuZeros(elemCount: Int): QuantizedType =
        when (this) {
            GgmlDType.f32 -> mutableListOf(Float.zeros(), elemCount)
            GgmlDType.f16 -> mutableListOf(zeros(), elemCount)
            GgmlDType.q40 -> mutableListOf(BlockQ40.zeros(), elemCount / BlockQ40.BLCK_SIZE)
            GgmlDType.q41 -> mutableListOf(BlockQ41.zeros(), elemCount / BlockQ41.BLCK_SIZE)
            GgmlDType.q50 -> mutableListOf(BlockQ50.zeros(), elemCount / BlockQ50.BLCK_SIZE)
            GgmlDType.q51 -> mutableListOf(BlockQ51.zeros(), elemCount / BlockQ51.BLCK_SIZE)
            GgmlDType.q80 -> mutableListOf(BlockQ80.zeros(), elemCount / BlockQ80.BLCK_SIZE)
            GgmlDType.q81 -> mutableListOf(BlockQ81.zeros(), elemCount / BlockQ81.BLCK_SIZE)
            GgmlDType.q2k -> mutableListOf(BlockQ2K.zeros(), elemCount / BlockQ2K.BLCK_SIZE)
            GgmlDType.q3k -> mutableListOf(BlockQ3K.zeros(), elemCount / BlockQ3K.BLCK_SIZE)
            GgmlDType.q4k -> mutableListOf(BlockQ4K.zeros(), elemCount / BlockQ4K.BLCK_SIZE)
            GgmlDType.q5k -> mutableListOf(BlockQ5K.zeros(), elemCount / BlockQ5K.BLCK_SIZE)
            GgmlDType.q6k -> mutableListOf(BlockQ6K.zeros(), elemCount / BlockQ6K.BLCK_SIZE)
            GgmlDType.q8k -> mutableListOf(BlockQ8K.zeros(), elemCount / BlockQ8K.BLCK_SIZE)
            GgmlDType.bf16 -> mutableListOf(zeros(), elemCount)
        }

    public fun fromData(data: Cow<List<UByte>>): QuantizedType {
        val data1: List<UByte> = data
        return when (this) {
            GgmlDType.f32 -> asTSlice<Float>(data1, typeLayoutT = PrimitiveLayouts.F32).toVec()
            GgmlDType.f16 -> asTSlice<F16>(data1, typeLayoutT = ObjectSlotLayout<F16>()).toVec()
            GgmlDType.q40 -> asTSlice<BlockQ40>(data1, typeLayoutT = ObjectSlotLayout<BlockQ40>()).toVec()
            GgmlDType.q41 -> asTSlice<BlockQ41>(data1, typeLayoutT = ObjectSlotLayout<BlockQ41>()).toVec()
            GgmlDType.q50 -> asTSlice<BlockQ50>(data1, typeLayoutT = ObjectSlotLayout<BlockQ50>()).toVec()
            GgmlDType.q51 -> asTSlice<BlockQ51>(data1, typeLayoutT = ObjectSlotLayout<BlockQ51>()).toVec()
            GgmlDType.q80 -> asTSlice<BlockQ80>(data1, typeLayoutT = ObjectSlotLayout<BlockQ80>()).toVec()
            GgmlDType.q81 -> asTSlice<BlockQ81>(data1, typeLayoutT = ObjectSlotLayout<BlockQ81>()).toVec()
            GgmlDType.q2k -> asTSlice<BlockQ2K>(data1, typeLayoutT = ObjectSlotLayout<BlockQ2K>()).toVec()
            GgmlDType.q3k -> asTSlice<BlockQ3K>(data1, typeLayoutT = ObjectSlotLayout<BlockQ3K>()).toVec()
            GgmlDType.q4k -> asTSlice<BlockQ4K>(data1, typeLayoutT = ObjectSlotLayout<BlockQ4K>()).toVec()
            GgmlDType.q5k -> asTSlice<BlockQ5K>(data1, typeLayoutT = ObjectSlotLayout<BlockQ5K>()).toVec()
            GgmlDType.q6k -> asTSlice<BlockQ6K>(data1, typeLayoutT = ObjectSlotLayout<BlockQ6K>()).toVec()
            GgmlDType.q8k -> asTSlice<BlockQ8K>(data1, typeLayoutT = ObjectSlotLayout<BlockQ8K>()).toVec()
            GgmlDType.bf16 -> asTSlice<Bf16>(data1, typeLayoutT = ObjectSlotLayout<Bf16>()).toVec()
        }
    }

    /**
     * The type size for blocks in bytes.
     */
    public fun typeSize(): Int =
        when (this) {
            GgmlDType.f32 -> 4
            GgmlDType.f16, GgmlDType.bf16 -> 2
            GgmlDType.q40 -> sizeOf<BlockQ40>(typeLayoutT = ObjectSlotLayout<BlockQ40>())
            GgmlDType.q41 -> sizeOf<BlockQ41>(typeLayoutT = ObjectSlotLayout<BlockQ41>())
            GgmlDType.q50 -> sizeOf<BlockQ50>(typeLayoutT = ObjectSlotLayout<BlockQ50>())
            GgmlDType.q51 -> sizeOf<BlockQ51>(typeLayoutT = ObjectSlotLayout<BlockQ51>())
            // https://github.com/ggerganov/llama.cpp/blob/468ea24fb4633a0d681f7ac84089566c1c6190cb/ggml.c#L932
            GgmlDType.q80 -> sizeOf<BlockQ80>(typeLayoutT = ObjectSlotLayout<BlockQ80>())
            GgmlDType.q81 -> sizeOf<BlockQ81>(typeLayoutT = ObjectSlotLayout<BlockQ81>())
            GgmlDType.q2k -> sizeOf<BlockQ2K>(typeLayoutT = ObjectSlotLayout<BlockQ2K>())
            GgmlDType.q3k -> sizeOf<BlockQ3K>(typeLayoutT = ObjectSlotLayout<BlockQ3K>())
            GgmlDType.q4k -> sizeOf<BlockQ4K>(typeLayoutT = ObjectSlotLayout<BlockQ4K>())
            GgmlDType.q5k -> sizeOf<BlockQ5K>(typeLayoutT = ObjectSlotLayout<BlockQ5K>())
            GgmlDType.q6k -> sizeOf<BlockQ6K>(typeLayoutT = ObjectSlotLayout<BlockQ6K>())
            GgmlDType.q8k -> sizeOf<BlockQ8K>(typeLayoutT = ObjectSlotLayout<BlockQ8K>())
        }

    /**
     * The block size, i.e. the number of elements stored in each block.
     */
    public fun blockSize(): Int =
        when (this) {
            GgmlDType.f32 -> 1
            GgmlDType.f16, GgmlDType.bf16 -> 1
            GgmlDType.q40 -> QK4_0
            GgmlDType.q41 -> QK4_1
            GgmlDType.q50 -> QK5_0
            GgmlDType.q51 -> QK5_1
            GgmlDType.q80 -> QK8_0
            GgmlDType.q81 -> QK8_1
            GgmlDType.q2k, GgmlDType.q3k, GgmlDType.q4k, GgmlDType.q5k, GgmlDType.q6k, GgmlDType.q8k -> QK_K
        }

    public companion object {
        internal fun fromU32(u: UInt): Result<GgmlDType> {
            val dtype =
                when (u) {
                    0 -> GgmlDType.F32
                    1 -> GgmlDType.F16
                    2 -> GgmlDType.Q4_0
                    3 -> GgmlDType.Q4_1
                    6 -> GgmlDType.Q5_0
                    7 -> GgmlDType.Q5_1
                    8 -> GgmlDType.Q8_0
                    9 -> GgmlDType.Q8_1
                    10 -> GgmlDType.Q2K
                    11 -> GgmlDType.Q3K
                    12 -> GgmlDType.Q4K
                    13 -> GgmlDType.Q5K
                    14 -> GgmlDType.Q6K
                    15 -> GgmlDType.Q8K
                    // https://github.com/ggerganov/ggml/blob/29d87fc6676e7ed0cdfdec0804b06001d9c2bb44/include/ggml.h#L389
                    30 -> GgmlDType.BF16
                    else -> bail("unknown dtype for tensor {u}")
                }
            return Ok(dtype)
        }
    }
}

// A version of GgmlType without `vecDot` so that it can be dynamic boxed.
public interface QuantizedType {
    public fun dtype(): GgmlDType

    public fun matmulT(mkn: Triple<Int, Int, Int>, lhs: List<Float>, dst: List<Float>): Result<Unit>

    public fun matmulTF16(mkn: Triple<Int, Int, Int>, lhs: List<F16>, dst: List<F16>): Result<Unit>

    public fun embedding(ids: List<UInt>, rows: Int, hidden: Int): Result<CpuStorage>

    public fun dequantize(elemCount: Int): Result<CpuStorage>

    public fun storageSizeInBytes(): Int

    public fun asPtr(): ConstPtr

    public fun blockSize(): Int

    public fun fromFloat(xs: List<Float>)

    public fun fromFloatImatrix(xs: List<Float>, imatrixWeights: List<Float>, nPerRow: Int)

    public fun size(): Int
}

public sealed class QMatMul : Module {
    public data class QTensor(
        public var value: QTensor,
    ) : QMatMul()
    public data class Tensor(
        public var value: Tensor,
    ) : QMatMul()
    public data class TensorF16(
        public var value: Tensor,
    ) : QMatMul()

    public fun dequantizeF16(): Result<Tensor> =
        when (this) {
            is QTensor -> {
                val (t) = this
                t.dequantizeF16(t.device())
            }
            is Tensor -> {
                val (t1) = this
                t1.toDtype(DType.F16)
            }
            is TensorF16 -> {
                val (t2) = this
                Ok(t2.clone())
            }
        }

    public fun forwardViaF16(xs: Tensor): Result<Tensor> {
        val w =
            when (val tryResult = this.dequantizeF16()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val inDtype = xs.dtype()
        val w1 =
            run {
                val matchVal = xs.dims()
                when {
                    matchVal.size == 4 -> {
                        val b1 = matchVal[0]
                        val b2 = matchVal[1]
                        when (
                            val tryResult =
                                when (val tryResult = w.broadcastLeft(Pair(b1, b2))) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }.t()
                        ) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    }
                    matchVal.size == 3 -> {
                        val bsize = matchVal[0]
                        when (
                            val tryResult =
                                when (val tryResult = w.broadcastLeft(bsize)) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }.t()
                        ) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    }
                    else ->
                        when (val tryResult = w.t()) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                }
            }
        return when (
            val tryResult =
                when (val tryResult = xs.toDtype(DType.F16)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }.matmul(w1)
        ) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }.toDtype(inDtype)
    }

    public fun indexedMoeForward(x: Tensor, ids: Tensor): Result<Tensor> =
        when (this) {
            is QTensor -> {
                val (t) = this
                t.indexedMoeForward(x, ids)
            }
            else -> {
                error("Not implemented!")
            }
        }

    public fun embedding(ids: Tensor): Result<Tensor> {
        return when (this) {
            is QTensor -> {
                val (t) = this
                t.embedding(ids)
            }
            is Tensor -> {
                val (w) = this
                var finalDims = ids.dims().toVec()
                finalDims.push(
                    when (val tryResult = w.dim(D.Minus1)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                )
                val ids1 =
                    when (
                        val tryResult =
                            when (val tryResult = ids.toDevice(w.device())) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }.flattenAll()
                    ) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                when (val tryResult = w.indexSelect(ids1, 0)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }.reshape(finalDims)
            }
            is TensorF16 -> {
                val (w1) = this
                var finalDims1 = ids.dims().toVec()
                finalDims1.push(
                    when (val tryResult = w1.dim(D.Minus1)) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    },
                )
                val ids2 =
                    when (
                        val tryResult =
                            when (val tryResult = ids.toDevice(w1.device())) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            }.flattenAll()
                    ) {
                        is Result.Ok -> tryResult.value
                        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                    }
                when (val tryResult = w1.indexSelect(ids2, 0)) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }.reshape(finalDims1)
            }
        }
    }

    public fun forward(xs: Tensor): Result<Tensor> {
        return when (this) {
            is QTensor -> {
                val (t) = this
                xs.applyOp1NoBwd(t.asRef())
            }
            is Tensor -> {
                val (w) = this
                val w1 =
                    run {
                        val matchVal = xs.dims()
                        when {
                            matchVal.size == 4 -> {
                                val b1 = matchVal[0]
                                val b2 = matchVal[1]
                                when (
                                    val tryResult =
                                        when (val tryResult = w.broadcastLeft(Pair(b1, b2))) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        }.t()
                                ) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                            }
                            matchVal.size == 3 -> {
                                val bsize = matchVal[0]
                                when (
                                    val tryResult =
                                        when (val tryResult = w.broadcastLeft(bsize)) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        }.t()
                                ) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                            }
                            else ->
                                when (val tryResult = w.t()) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                        }
                    }
                xs.matmul(w1)
            }
            is TensorF16 -> {
                val (w2) = this
                val inDtype = xs.dtype()
                val w3 =
                    run {
                        val matchVal1 = xs.dims()
                        when {
                            matchVal1.size == 4 -> {
                                val b11 = matchVal1[0]
                                val b21 = matchVal1[1]
                                when (
                                    val tryResult =
                                        when (val tryResult = w2.broadcastLeft(Pair(b11, b21))) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        }.t()
                                ) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                            }
                            matchVal1.size == 3 -> {
                                val bsize1 = matchVal1[0]
                                when (
                                    val tryResult =
                                        when (val tryResult = w2.broadcastLeft(bsize1)) {
                                            is Result.Ok -> tryResult.value
                                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                        }.t()
                                ) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                            }
                            else ->
                                when (val tryResult = w2.t()) {
                                    is Result.Ok -> tryResult.value
                                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                                }
                        }
                    }
                when (
                    val tryResult =
                        when (val tryResult = xs.toDtype(DType.F16)) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }.matmul(w3)
                ) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }.toDtype(inDtype)
            }
        }
    }

    public companion object {
        public fun fromArc(qtensor: QTensor): Result<QMatMul> {
            val dequantize =
                when (qtensor.dtype()) {
                    GgmlDType.f32, GgmlDType.f16, GgmlDType.bf16 -> true
                    else -> DEQUANTIZE_ALL.with({ b -> b })
                }
            val t =
                if (dequantize) {
                    val tensor =
                        when (val tryResult = qtensor.dequantize(qtensor.device())) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    QMatMul.Tensor(tensor)
                } else if (DEQUANTIZE_ALL_F16.with({ b -> b })) {
                    val tensor1 =
                        when (val tryResult = qtensor.dequantizeF16(qtensor.device())) {
                            is Result.Ok -> tryResult.value
                            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                        }
                    QMatMul.TensorF16(tensor1)
                } else {
                    QMatMul.QTensor(qtensor)
                }
            return Ok(t)
        }

        public fun fromQtensor(qtensor: QTensor): Result<QMatMul> = QMatMul.fromArc(qtensor)
    }
}
