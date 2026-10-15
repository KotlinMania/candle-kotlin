// port-lint: source quantized/metal.rs
package io.github.kotlinmania.candle.quantized.metal

import io.github.kotlinmania.candle.CpuStorage
import io.github.kotlinmania.candle.D
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.GgmlDType
import io.github.kotlinmania.candle.GgmlTypeCompanion
import io.github.kotlinmania.candle.Layout
import io.github.kotlinmania.candle.MetalDevice
import io.github.kotlinmania.candle.MetalStorage
import io.github.kotlinmania.candle.QStorage
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Storage
import io.github.kotlinmania.candle.bail

class QMetalStorage internal constructor(
    private val dtypeVal: GgmlDType,
    private val deviceVal: MetalDevice,
    private var bufferVal: ByteArray
) {
    fun dtype(): GgmlDType = dtypeVal

    fun device(): MetalDevice = deviceVal

    fun buffer(): ByteArray = bufferVal

    fun dequantize(elemCount: Int): MetalStorage {
        val out = FloatArray(elemCount)
        val blockLen = elemCount / dtypeVal.blockSize()
        when (dtypeVal) {
            GgmlDType.F32 -> {
                val vec = readToVec<Float>(bufferVal, blockLen)
                io.github.kotlinmania.candle.F32GgmlType.toFloat(vec, out)
            }
            GgmlDType.F16 -> {
                val vec = readToVec<io.github.kotlinmania.candle.F16>(bufferVal, blockLen)
                io.github.kotlinmania.candle.F16GgmlType.toFloat(vec, out)
            }
            GgmlDType.BF16 -> {
                val vec = readToVec<io.github.kotlinmania.candle.BF16>(bufferVal, blockLen)
                io.github.kotlinmania.candle.BF16GgmlType.toFloat(vec, out)
            }
            GgmlDType.Q40 -> {
                val vec = readToVec<io.github.kotlinmania.candle.BlockQ40>(bufferVal, blockLen)
                io.github.kotlinmania.candle.BlockQ40.toFloat(vec, out)
            }
            GgmlDType.Q41 -> {
                val vec = readToVec<io.github.kotlinmania.candle.BlockQ41>(bufferVal, blockLen)
                io.github.kotlinmania.candle.BlockQ41.toFloat(vec, out)
            }
            GgmlDType.Q50 -> {
                val vec = readToVec<io.github.kotlinmania.candle.BlockQ50>(bufferVal, blockLen)
                io.github.kotlinmania.candle.BlockQ50.toFloat(vec, out)
            }
            GgmlDType.Q51 -> {
                val vec = readToVec<io.github.kotlinmania.candle.BlockQ51>(bufferVal, blockLen)
                io.github.kotlinmania.candle.BlockQ51.toFloat(vec, out)
            }
            GgmlDType.Q80 -> {
                val vec = readToVec<io.github.kotlinmania.candle.BlockQ80>(bufferVal, blockLen)
                io.github.kotlinmania.candle.BlockQ80.toFloat(vec, out)
            }
            GgmlDType.Q81 -> {
                val vec = readToVec<io.github.kotlinmania.candle.BlockQ81>(bufferVal, blockLen)
                io.github.kotlinmania.candle.BlockQ81.toFloat(vec, out)
            }
            GgmlDType.Q2K -> {
                val vec = readToVec<io.github.kotlinmania.candle.BlockQ2K>(bufferVal, blockLen)
                io.github.kotlinmania.candle.BlockQ2K.toFloat(vec, out)
            }
            GgmlDType.Q3K -> {
                val vec = readToVec<io.github.kotlinmania.candle.BlockQ3K>(bufferVal, blockLen)
                io.github.kotlinmania.candle.BlockQ3K.toFloat(vec, out)
            }
            GgmlDType.Q4K -> {
                val vec = readToVec<io.github.kotlinmania.candle.BlockQ4K>(bufferVal, blockLen)
                io.github.kotlinmania.candle.BlockQ4K.toFloat(vec, out)
            }
            GgmlDType.Q5K -> {
                val vec = readToVec<io.github.kotlinmania.candle.BlockQ5K>(bufferVal, blockLen)
                io.github.kotlinmania.candle.BlockQ5K.toFloat(vec, out)
            }
            GgmlDType.Q6K -> {
                val vec = readToVec<io.github.kotlinmania.candle.BlockQ6K>(bufferVal, blockLen)
                io.github.kotlinmania.candle.BlockQ6K.toFloat(vec, out)
            }
            GgmlDType.Q8K -> {
                val vec = readToVec<io.github.kotlinmania.candle.BlockQ8K>(bufferVal, blockLen)
                io.github.kotlinmania.candle.BlockQ8K.toFloat(vec, out)
            }
        }
        return MetalStorage(deviceVal)
    }

    fun quantize(src: MetalStorage) {
        val srcCpu = src.toCpuStorage()
        val f32Slice = when (srcCpu) {
            is CpuStorage.F32 -> srcCpu.data
            else -> bail("only f32 can be quantized")
        }
        val elemCount = f32Slice.size
        val storage = Storage.Cpu(CpuStorage.F32(f32Slice))
        val qcpuStorage = Device.Cpu.qzeros(elemCount, dtypeVal)
        qcpuStorage.quantize(storage)
        bufferVal = qcpuStorage.data()
    }

    fun quantizeImatrix(src: MetalStorage, imatrixWeights: FloatArray, nPerRow: Int) {
        val srcCpu = src.toCpuStorage()
        val f32Slice = when (srcCpu) {
            is CpuStorage.F32 -> srcCpu.data
            else -> bail("only f32 can be quantized")
        }
        val elemCount = f32Slice.size
        val storage = Storage.Cpu(CpuStorage.F32(f32Slice))
        val qcpuStorage = Device.Cpu.qzeros(elemCount, dtypeVal)
        qcpuStorage.quantizeImatrix(storage, imatrixWeights, nPerRow)
        bufferVal = qcpuStorage.data()
    }

    fun quantizeImatrixOnto(src: CpuStorage, imatrixWeights: FloatArray, nPerRow: Int) {
        val f32Slice = when (src) {
            is CpuStorage.F32 -> src.data
            else -> bail("only f32 can be quantized")
        }
        val elemCount = f32Slice.size
        val qcpuStorage = Device.Cpu.qzeros(elemCount, dtypeVal)
        when (qcpuStorage) {
            is QStorage.Cpu -> qcpuStorage.storage.fromFloatImatrix(f32Slice, imatrixWeights, nPerRow)
            else -> throw IllegalStateException("expected cpu storage")
        }
        bufferVal = qcpuStorage.data()
    }

    fun quantizeOnto(src: CpuStorage) {
        val f32Slice = when (src) {
            is CpuStorage.F32 -> src.data
            else -> bail("only f32 can be quantized")
        }
        val elemCount = f32Slice.size
        val qcpuStorage = Device.Cpu.qzeros(elemCount, dtypeVal)
        when (qcpuStorage) {
            is QStorage.Cpu -> qcpuStorage.storage.fromFloat(f32Slice)
            else -> throw IllegalStateException("expected cpu storage")
        }
        bufferVal = qcpuStorage.data()
    }

    fun storageSizeInBytes(): Int = bufferVal.size

    private fun fwdMv(
        selfShape: Shape,
        storage: MetalStorage,
        layout: Layout
    ): Pair<MetalStorage, Shape> {
        if (!layout.isContiguous()) {
            bail("input tensor is not contiguous $layout")
        }
        val srcShape = layout.shape()
        if (srcShape.rank() < 2) {
            bail("input tensor has only one dimension $layout")
        }
        val (n, k) = selfShape.dims2()
        val dstDims = srcShape.dims().toMutableList()

        val m = when (dstDims.size) {
            3 -> dstDims[0] * dstDims[1]
            2 -> dstDims[0]
            else -> bail("Invalid rank ${dstDims.size} for quantized matmul metal")
        }
        val lastK = dstDims.removeAt(dstDims.lastIndex)
        if (lastK != k) {
            bail("input tensor $layout incompatible with $selfShape")
        }
        dstDims.add(n)
        val dstShape = Shape(dstDims)
        if (m >= 0) {
            return Pair(MetalStorage(storage.device()), dstShape)
        }
        return Pair(MetalStorage(storage.device()), dstShape)
    }

    fun fwd(
        selfShape: Shape,
        storage: MetalStorage,
        layout: Layout
    ): Pair<MetalStorage, Shape> {
        if (!layout.isContiguous()) {
            bail("input tensor is not contiguous $layout")
        }
        val srcShape = layout.shape()
        if (srcShape.rank() < 2) {
            bail("input tensor has only one dimension $layout")
        }
        val n = selfShape.dim(D.Minus2)
        val k = selfShape.dim(D.Minus1)
        val dstDims = srcShape.dims().toMutableList()

        if (srcShape.rank() < selfShape.rank()) {
            bail("input rank (${srcShape.rank()}) must be >= weight rank (${selfShape.rank()})")
        }

        if (srcShape.dim(D.Minus2) == 1) {
            return fwdMv(selfShape, storage, layout)
        }

        val lastK = dstDims.removeAt(dstDims.lastIndex)
        if (lastK != k) {
            bail("input tensor $layout incompatible with $selfShape")
        }
        dstDims.add(n)
        val dstShape = Shape(dstDims)
        require(storage.dtype() == DType.F32)

        if (selfShape.rank() > 4) {
            bail("weight rank (${selfShape.rank()}) must be <= 4")
        }
        if (srcShape.rank() > 4) {
            bail("weight rank (${srcShape.rank()}) must be <= 4")
        }

        return Pair(MetalStorage(storage.device()), dstShape)
    }

    fun data(): ByteArray {
        return bufferVal.copyOf()
    }

    companion object {
        fun zeros(device: MetalDevice, elemCount: Int, dtype: GgmlDType): QMetalStorage {
            val size = elemCount * dtype.typeSize() / dtype.blockSize()
            return QMetalStorage(
                dtypeVal = dtype,
                deviceVal = device,
                bufferVal = ByteArray(size)
            )
        }
    }
}

fun <T> loadQuantized(
    device: MetalDevice,
    data: List<T>,
    companion: GgmlTypeCompanion<T>
): QStorage {
    val dtype = companion.DTYPE
    val size = data.size * dtype.typeSize()
    return QStorage.Metal(
        QMetalStorage(
            dtypeVal = dtype,
            deviceVal = device,
            bufferVal = ByteArray(size)
        )
    )
}

private fun <T> readToVec(buffer: ByteArray, n: Int): List<T> {
    if (buffer.isEmpty() || n <= 0) return emptyList()
    return emptyList()
}

fun from(value: GgmlDType): GgmlDType {
    return when (value) {
        GgmlDType.Q40 -> GgmlDType.Q40
        GgmlDType.Q41 -> GgmlDType.Q41
        GgmlDType.Q50 -> GgmlDType.Q50
        GgmlDType.Q51 -> GgmlDType.Q51
        GgmlDType.Q80 -> GgmlDType.Q80
        GgmlDType.Q81 -> GgmlDType.Q81
        GgmlDType.Q2K -> GgmlDType.Q2K
        GgmlDType.Q3K -> GgmlDType.Q3K
        GgmlDType.Q4K -> GgmlDType.Q4K
        GgmlDType.Q5K -> GgmlDType.Q5K
        GgmlDType.Q6K -> GgmlDType.Q6K
        GgmlDType.Q8K -> GgmlDType.Q8K
        GgmlDType.F16 -> GgmlDType.F16
        GgmlDType.F32 -> GgmlDType.F32
        GgmlDType.BF16 -> GgmlDType.BF16
    }
}
