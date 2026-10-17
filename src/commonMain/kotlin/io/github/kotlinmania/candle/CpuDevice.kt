package io.github.kotlinmania.candle

import io.github.kotlinmania.rand.distr.Uniform
import io.github.kotlinmania.rand.rng
import kotlin.math.PI
import kotlin.math.cos
import kotlin.math.ln
import kotlin.math.sin
import kotlin.math.sqrt

public object CpuDevice {
    public fun location(): DeviceLocation = DeviceLocation.Cpu

    public fun sameDevice(other: CpuDevice): Boolean = true

    public fun sameDevice(other: BackendDevice): Boolean = false

    public fun storageFromSlice(s: FloatArray): CpuStorage = CpuStorage.F32(s.copyOf())

    public fun storageFromSlice(s: DoubleArray): CpuStorage = CpuStorage.F64(s.copyOf())

    public fun storageFromSlice(s: IntArray): CpuStorage = CpuStorage.I32(s.copyOf())

    public fun storageFromSlice(s: LongArray): CpuStorage = CpuStorage.I64(s.copyOf())

    public fun storageFromSlice(s: ShortArray): CpuStorage = CpuStorage.I16(s.copyOf())

    public fun storageFromSlice(s: UIntArray): CpuStorage = CpuStorage.U32(s.copyOf())

    public fun storageFromSlice(s: UByteArray): CpuStorage = CpuStorage.U8(s.copyOf())

    public fun storageFromCpuStorage(storage: CpuStorage): CpuStorage = storage

    public fun storageFromCpuStorageOwned(storage: CpuStorage): CpuStorage = storage

    public fun new(ordinal: Int = 0): CpuDevice = this

    public fun setSeed(seed: Long) {
        bail("cannot seed the CPU rng with set_seed")
    }

    public fun getCurrentSeed(): Long {
        bail("cannot get the CPU rng seed with get_current_seed")
    }

    public fun randUniform(
        shape: Shape,
        dtype: DType,
        min: Double,
        max: Double,
    ): CpuStorage {
        val elemCount = shape.elemCount()
        val rng = rng()
        return when (dtype) {
            DType.BF16 -> {
                val uniform = Uniform.new(min.toFloat(), max.toFloat()).map { it }.unwrap()
                val data =
                    ShortArray(elemCount) {
                        Float16Utils.floatToBF16Bits(uniform.sample(rng))
                    }
                CpuStorage.BF16(data)
            }

            DType.F16 -> {
                val uniform = Uniform.new(min.toFloat(), max.toFloat()).map { it }.unwrap()
                val data =
                    ShortArray(elemCount) {
                        Float16Utils.floatToF16Bits(uniform.sample(rng))
                    }
                CpuStorage.F16(data)
            }

            DType.F32 -> {
                val uniform = Uniform.new(min.toFloat(), max.toFloat()).map { it }.unwrap()
                val data =
                    FloatArray(elemCount) {
                        uniform.sample(rng)
                    }
                CpuStorage.F32(data)
            }

            DType.F64 -> {
                val uniform = Uniform.new(min, max).map { it }.unwrap()
                val data =
                    DoubleArray(elemCount) {
                        uniform.sample(rng)
                    }
                CpuStorage.F64(data)
            }

            DType.F8E4M3 -> {
                val uniform = Uniform.new(min, max).map { it }.unwrap()
                val data =
                    ByteArray(elemCount) {
                        uniform.sample(rng).toInt().toByte()
                    }
                CpuStorage.F8E4M3(data)
            }

            else -> {
                bail("Unsupported dtype for randUniform $dtype")
            }
        }
    }

    public fun randNormal(
        shape: Shape,
        dtype: DType,
        mean: Double,
        std: Double,
    ): CpuStorage {
        val elemCount = shape.elemCount()
        val rng = rng()
        val normalValues = DoubleArray(elemCount)
        var i = 0
        while (i < elemCount) {
            val u1 = rng.randomDouble().coerceAtLeast(1e-15)
            val u2 = rng.randomDouble()
            val z0 = sqrt(-2.0 * ln(u1)) * cos(2.0 * PI * u2)
            val z1 = sqrt(-2.0 * ln(u1)) * sin(2.0 * PI * u2)
            normalValues[i++] = mean + std * z0
            if (i < elemCount) {
                normalValues[i++] = mean + std * z1
            }
        }
        return when (dtype) {
            DType.BF16 -> {
                val data = ShortArray(elemCount) { Float16Utils.floatToBF16Bits(normalValues[it].toFloat()) }
                CpuStorage.BF16(data)
            }

            DType.F16 -> {
                val data = ShortArray(elemCount) { Float16Utils.floatToF16Bits(normalValues[it].toFloat()) }
                CpuStorage.F16(data)
            }

            DType.F32 -> {
                val data = FloatArray(elemCount) { normalValues[it].toFloat() }
                CpuStorage.F32(data)
            }

            DType.F64 -> {
                CpuStorage.F64(normalValues)
            }

            DType.F8E4M3 -> {
                val data = ByteArray(elemCount) { normalValues[it].toInt().toByte() }
                CpuStorage.F8E4M3(data)
            }

            else -> {
                bail("Unsupported dtype for randNormal $dtype")
            }
        }
    }

    public fun allocUninit(
        shape: Shape,
        dtype: DType,
    ): CpuStorage = zerosImpl(shape, dtype)

    public fun zerosImpl(
        shape: Shape,
        dtype: DType,
    ): CpuStorage {
        val elemCount = shape.elemCount()
        return when (dtype) {
            DType.U8 -> CpuStorage.U8(UByteArray(elemCount))
            DType.U32 -> CpuStorage.U32(UIntArray(elemCount))
            DType.I16 -> CpuStorage.I16(ShortArray(elemCount))
            DType.I32 -> CpuStorage.I32(IntArray(elemCount))
            DType.I64 -> CpuStorage.I64(LongArray(elemCount))
            DType.BF16 -> CpuStorage.BF16(ShortArray(elemCount))
            DType.F16 -> CpuStorage.F16(ShortArray(elemCount))
            DType.F32 -> CpuStorage.F32(FloatArray(elemCount))
            DType.F64 -> CpuStorage.F64(DoubleArray(elemCount))
            DType.F8E4M3 -> CpuStorage.F8E4M3(ByteArray(elemCount))
            else -> bail("Unsupported dtype for zeros $dtype")
        }
    }

    public fun synchronize(): Unit = Unit
}

private fun io.github.kotlinmania.rand.rngs.ThreadRng.randomDouble(): Double =
    (nextU64() shr 11).toDouble() * (1.0 / (1uL shl 53).toDouble())
