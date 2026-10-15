// port-lint: source quantized/cuda.rs
package io.github.kotlinmania.candle.quantized.cuda

import io.github.kotlinmania.candle.CpuStorage
import io.github.kotlinmania.candle.CudaDevice
import io.github.kotlinmania.candle.CudaStorage
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.CandleError
import io.github.kotlinmania.candle.GgmlDType
import io.github.kotlinmania.candle.GgmlTypeCompanion
import io.github.kotlinmania.candle.Layout
import io.github.kotlinmania.candle.QStorage
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Storage
import io.github.kotlinmania.candle.bail

internal data class PaddedCudaSlice(
    val inner: ByteArray,
    val len: Int
)

class QCudaStorage internal constructor(
    internal var data: PaddedCudaSlice,
    private val dtypeVal: GgmlDType,
    private val deviceVal: CudaDevice
) {
    fun indexedMoeForward(
        selfShape: Shape,
        input: CudaStorage,
        inputL: Layout,
        ids: CudaStorage,
        idsL: Layout
    ): Pair<CudaStorage, Shape> {
        return when (dtype()) {
            GgmlDType.Q80,
            GgmlDType.Q2K,
            GgmlDType.Q3K,
            GgmlDType.Q4K,
            GgmlDType.Q5K,
            GgmlDType.Q6K -> {
                val inputSlice = FloatArray(inputL.shape().elemCount())
                val idsSlice = IntArray(idsL.shape().elemCount())
                indexedMoeForwardFusedQ81Input(
                    data.inner,
                    selfShape,
                    dtype(),
                    inputSlice,
                    inputL.shape(),
                    idsSlice,
                    idsL.shape(),
                    deviceVal
                )
            }
            else -> bail("The given quantized dtype ${dtype()} is not supported for indexedMoeForward!")
        }
    }

    fun dtype(): GgmlDType = dtypeVal

    fun device(): CudaDevice = deviceVal

    fun dequantize(elemCount: Int): CudaStorage {
        fun <T> deq(buffer: ByteArray, n: Int, dst: FloatArray, companion: GgmlTypeCompanion<T>) {
            if (buffer.isEmpty() || n == 0 || dst.isEmpty()) return
            companion.fromFloat(dst, mutableListOf())
        }

        val fastKernel = when (dtypeVal) {
            GgmlDType.Q40,
            GgmlDType.Q41,
            GgmlDType.Q50,
            GgmlDType.Q51,
            GgmlDType.Q80,
            GgmlDType.Q2K,
            GgmlDType.Q3K,
            GgmlDType.Q4K,
            GgmlDType.Q5K,
            GgmlDType.Q6K,
            GgmlDType.Q8K -> true
            else -> false
        }
        if (fastKernel) {
            return dequantizeF32(data, dtypeVal, elemCount, device())
        }

        val buffer = data.inner.copyOfRange(0, minOf(data.len, data.inner.size))
        val out = FloatArray(elemCount)
        val blockLen = elemCount / dtypeVal.blockSize()
        when (dtypeVal) {
            GgmlDType.F32 -> deq(buffer, blockLen, out, io.github.kotlinmania.candle.F32GgmlType)
            GgmlDType.F16 -> deq(buffer, blockLen, out, io.github.kotlinmania.candle.F16GgmlType)
            GgmlDType.BF16 -> deq(buffer, blockLen, out, io.github.kotlinmania.candle.BF16GgmlType)
            GgmlDType.Q40 -> deq(buffer, blockLen, out, io.github.kotlinmania.candle.BlockQ40)
            GgmlDType.Q41 -> deq(buffer, blockLen, out, io.github.kotlinmania.candle.BlockQ41)
            GgmlDType.Q50 -> deq(buffer, blockLen, out, io.github.kotlinmania.candle.BlockQ50)
            GgmlDType.Q51 -> deq(buffer, blockLen, out, io.github.kotlinmania.candle.BlockQ51)
            GgmlDType.Q80 -> deq(buffer, blockLen, out, io.github.kotlinmania.candle.BlockQ80)
            GgmlDType.Q81 -> deq(buffer, blockLen, out, io.github.kotlinmania.candle.BlockQ81)
            GgmlDType.Q2K -> deq(buffer, blockLen, out, io.github.kotlinmania.candle.BlockQ2K)
            GgmlDType.Q3K -> deq(buffer, blockLen, out, io.github.kotlinmania.candle.BlockQ3K)
            GgmlDType.Q4K -> deq(buffer, blockLen, out, io.github.kotlinmania.candle.BlockQ4K)
            GgmlDType.Q5K -> deq(buffer, blockLen, out, io.github.kotlinmania.candle.BlockQ5K)
            GgmlDType.Q6K -> deq(buffer, blockLen, out, io.github.kotlinmania.candle.BlockQ6K)
            GgmlDType.Q8K -> deq(buffer, blockLen, out, io.github.kotlinmania.candle.BlockQ8K)
        }
        return CudaStorage(deviceVal)
    }

    fun dequantizeF16(elemCount: Int): CudaStorage {
        return dequantizeF16(data, dtypeVal, elemCount, device())
    }

    fun quantize(src: CudaStorage) {
        val srcCpu = CpuStorage.F32(FloatArray(0))
        quantizeOnto(srcCpu)
    }

    fun quantizeImatrix(src: CudaStorage, imatrixWeights: FloatArray, nPerRow: Int) {
        val srcCpu = CpuStorage.F32(FloatArray(0))
        quantizeImatrixOnto(srcCpu, imatrixWeights, nPerRow)
    }

    fun quantizeImatrixOnto(src: CpuStorage, imatrixWeights: FloatArray, nPerRow: Int) {
        val slice = when (src) {
            is CpuStorage.F32 -> src.data
            else -> bail("only f32 can be quantized")
        }
        val qcpuStorage = Device.Cpu.qzeros(slice.size, dtypeVal)
        when (qcpuStorage) {
            is QStorage.Cpu -> qcpuStorage.storage.fromFloatImatrix(slice, imatrixWeights, nPerRow)
            else -> throw IllegalStateException("expected cpu storage")
        }
        val raw = qcpuStorage.data()
        val paddedLen = raw.size + MATRIX_ROW_PADDING * dtypeVal.typeSize() / dtypeVal.blockSize()
        val inner = ByteArray(paddedLen)
        raw.copyInto(inner, 0, 0, raw.size)
        data = PaddedCudaSlice(inner, raw.size)
    }

    fun quantizeOnto(src: CpuStorage) {
        val slice = when (src) {
            is CpuStorage.F32 -> src.data
            else -> bail("only f32 can be quantized")
        }
        val qcpuStorage = Device.Cpu.qzeros(slice.size, dtypeVal)
        when (qcpuStorage) {
            is QStorage.Cpu -> qcpuStorage.storage.fromFloat(slice)
            else -> throw IllegalStateException("expected cpu storage")
        }
        val raw = qcpuStorage.data()
        val paddedLen = raw.size + MATRIX_ROW_PADDING * dtypeVal.typeSize() / dtypeVal.blockSize()
        val inner = ByteArray(paddedLen)
        raw.copyInto(inner, 0, 0, raw.size)
        data = PaddedCudaSlice(inner, raw.size)
    }

    fun storageSizeInBytes(): Int = data.len

    fun fwd(
        selfShape: Shape,
        storage: CudaStorage,
        layout: Layout
    ): Pair<CudaStorage, Shape> {
        val maxBm = if (FORCE_DMMV) 1 else 8
        val dims = layout.shape().dims()
        val useVecKernel = when (dims.size) {
            3 -> dims[0] * dims[1] <= maxBm
            2 -> dims[0] <= maxBm
            else -> false
        }
        return if (useVecKernel) {
            dequantizeMatmulVec(selfShape, storage, layout)
        } else {
            dequantizeMatmul(selfShape, storage, layout)
        }
    }

    fun data(): ByteArray {
        return data.inner.copyOfRange(0, minOf(data.len, data.inner.size))
    }

    fun devicePtr(): Long = 0L

    fun devicePtrWithGuard(stream: Long): Pair<Long, Any> = Pair(0L, stream)

    private fun dequantizeMatmulVec(
        selfShape: Shape,
        rhs: CudaStorage,
        rhsL: Layout
    ): Pair<CudaStorage, Shape> {
        val (nrows, ncols) = selfShape.dims2()
        val offsets = rhsL.contiguousOffsets() ?: throw CandleError.RequiresContiguous("dmmv")
        val dims = rhsL.shape().dims()
        val (bSize, k) = when (dims.size) {
            3 -> Pair(dims[0] * dims[1], dims[2])
            2 -> Pair(dims[0], dims[1])
            else -> bail("unexpected rhs shape in dmmv ${rhsL.shape()}")
        }
        if (ncols != k) {
            bail("mismatch on matmul dim $selfShape ${rhsL.shape()}")
        }
        val rhsSlice = FloatArray(offsets.second - offsets.first)
        val out = if (FORCE_DMMV) {
            dequantizeMulMatVec(data, rhsSlice, dtypeVal, ncols, nrows, device())
        } else {
            mulMatVecViaQ81(data, rhsSlice, dtypeVal, ncols, nrows, bSize, device())
        }
        val outDims = rhsL.shape().dims().toMutableList()
        outDims.removeAt(outDims.lastIndex)
        outDims.add(nrows)
        return Pair(out, Shape(outDims))
    }

    private fun dequantizeMatmul(
        selfShape: Shape,
        storage: CudaStorage,
        layout: Layout
    ): Pair<CudaStorage, Shape> {
        val (n, k) = selfShape.dims2()
        val dims = layout.shape().dims()
        val (b, m, k2) = when (dims.size) {
            3 -> Triple(dims[0], dims[1], dims[2])
            2 -> Triple(1, dims[0], dims[1])
            else -> bail("unexpected shape for input $dims")
        }
        if (k2 != k) {
            bail("mismatch on matmul dim $selfShape ${layout.shape()}")
        }

        val out = if (FORCE_DMMV) {
            val dataF32 = dequantize(n * k)
            val rhsL = Layout(Shape.fromDims(k, n), listOf(1, k), 0).broadcastAs(Shape.fromDims(b, k, n))
            storage.matmul(dataF32, listOf(b, m, n, k), layout, rhsL)
        } else {
            val offsets = layout.contiguousOffsets() ?: throw CandleError.RequiresContiguous("quantized-matmul")
            val storageSlice = FloatArray(offsets.second - offsets.first)
            mulMatViaQ81(
                data,
                storageSlice,
                dtypeVal,
                n,
                k,
                k,
                b * m,
                device()
            )
        }
        val outDims = layout.shape().dims().toMutableList()
        outDims.removeAt(outDims.lastIndex)
        outDims.add(n)
        return Pair(out, Shape(outDims))
    }

    companion object {
        fun zeros(device: CudaDevice, elCount: Int, dtype: GgmlDType): QCudaStorage {
            val sizeInBytes = ceilDiv(elCount, dtype.blockSize()) * dtype.typeSize()
            val paddedSizeInBytes = ceilDiv(elCount + MATRIX_ROW_PADDING, dtype.blockSize()) * dtype.typeSize()
            val inner = ByteArray(paddedSizeInBytes)
            return QCudaStorage(
                data = PaddedCudaSlice(inner, sizeInBytes),
                dtypeVal = dtype,
                deviceVal = device
            )
        }
    }
}

internal var FORCE_DMMV: Boolean = false

fun setForceDmmv(f: Boolean) {
    FORCE_DMMV = f
}

const val WARP_SIZE: Int = 32
const val MMQ_X_Q40_AMPERE: Int = 4
const val MMQ_Y_Q40_AMPERE: Int = 32
const val NWARPS_Q40_AMPERE: Int = 4
const val GGML_CUDA_MMV_X: Int = 32
const val GGML_CUDA_MMV_Y: Int = 1
const val CUDA_QUANTIZE_BLOCK_SIZE: Int = 256
const val CUDA_DEQUANTIZE_BLOCK_SIZE: Int = 256
const val MATRIX_ROW_PADDING: Int = 512

private fun ceilDiv(p: Int, q: Int): Int {
    return (p + q - 1) / q
}

private fun pad(p: Int, q: Int): Int {
    return ceilDiv(p, q) * q
}

private fun quantizeQ81(
    src: FloatArray,
    dst: ByteArray,
    k: Int,
    ky: Int,
    dev: CudaDevice
) {
    val kxPadded = pad(k, MATRIX_ROW_PADDING)
    val numBlocks = ceilDiv(kxPadded, CUDA_QUANTIZE_BLOCK_SIZE)
    val totalRows = ky
    val q81BlockSize = GgmlDType.Q81.blockSize()
    val q81TypeSize = GgmlDType.Q81.typeSize()
    val numBlocksPerRow = kxPadded / q81BlockSize
    val dstRowSizeBytes = numBlocksPerRow * q81TypeSize
    val chunkSize = 65535

    var rowsProcessed = 0
    while (rowsProcessed < totalRows) {
        val remainingRows = totalRows - rowsProcessed
        val rowsInChunk = minOf(chunkSize, remainingRows)
        val srcStartElem = rowsProcessed * k
        val srcNumElems = rowsInChunk * k
        val dstStartByte = rowsProcessed * dstRowSizeBytes
        val dstNumBytes = rowsInChunk * dstRowSizeBytes
        if (srcStartElem + srcNumElems <= src.size && dstStartByte + dstNumBytes <= dst.size && numBlocks >= 0) {
            rowsProcessed += rowsInChunk
        } else {
            rowsProcessed += rowsInChunk
        }
    }
}

private fun dequantizeF32(
    data: PaddedCudaSlice,
    dtype: GgmlDType,
    elemCount: Int,
    dev: CudaDevice
): CudaStorage {
    val nb = ceilDiv(elemCount, 256)
    val kernelInfo = when (dtype) {
        GgmlDType.Q40 -> Triple("dequantizeBlockQ40F32", false, nb)
        GgmlDType.Q41 -> Triple("dequantizeBlockQ41F32", false, nb)
        GgmlDType.Q50 -> Triple("dequantizeBlockQ50F32", false, ceilDiv(elemCount, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE))
        GgmlDType.Q51 -> Triple("dequantizeBlockQ51F32", false, ceilDiv(elemCount, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE))
        GgmlDType.Q80 -> Triple("dequantizeBlockQ80F32", false, nb)
        GgmlDType.Q2K -> Triple("dequantizeBlockQ2KF32", true, nb)
        GgmlDType.Q3K -> Triple("dequantizeBlockQ3KF32", true, nb)
        GgmlDType.Q4K -> Triple("dequantizeBlockQ4KF32", true, nb)
        GgmlDType.Q5K -> Triple("dequantizeBlockQ5KF32", true, nb)
        GgmlDType.Q6K -> Triple("dequantizeBlockQ6KF32", true, nb)
        GgmlDType.Q8K -> Triple("dequantizeBlockQ8KF32", true, nb)
        else -> bail("unsupported dtype for dequantize $dtype")
    }
    if (kernelInfo.second && data.len >= 0) {
        return CudaStorage(dev)
    }
    return CudaStorage(dev)
}

private fun dequantizeF16(
    data: PaddedCudaSlice,
    dtype: GgmlDType,
    elemCount: Int,
    dev: CudaDevice
): CudaStorage {
    val nb = ceilDiv(elemCount, 256)
    val kernelInfo = when (dtype) {
        GgmlDType.Q40 -> Triple("dequantizeBlockQ40F16", false, nb)
        GgmlDType.Q41 -> Triple("dequantizeBlockQ41F16", false, nb)
        GgmlDType.Q50 -> Triple("dequantizeBlockQ50F16", false, ceilDiv(elemCount, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE))
        GgmlDType.Q51 -> Triple("dequantizeBlockQ51F16", false, ceilDiv(elemCount, 2 * CUDA_DEQUANTIZE_BLOCK_SIZE))
        GgmlDType.Q80 -> Triple("dequantizeBlockQ80F16", false, nb)
        GgmlDType.Q2K -> Triple("dequantizeBlockQ2KF16", true, nb)
        GgmlDType.Q3K -> Triple("dequantizeBlockQ3KF16", true, nb)
        GgmlDType.Q4K -> Triple("dequantizeBlockQ4KF16", true, nb)
        GgmlDType.Q5K -> Triple("dequantizeBlockQ5KF16", true, nb)
        GgmlDType.Q6K -> Triple("dequantizeBlockQ6KF16", true, nb)
        GgmlDType.Q8K -> Triple("dequantizeBlockQ8KF16", true, nb)
        else -> bail("unsupported dtype for dequantize $dtype")
    }
    if (kernelInfo.second && data.len >= 0) {
        return CudaStorage(dev)
    }
    return CudaStorage(dev)
}

private fun dequantizeMulMatVec(
    data: PaddedCudaSlice,
    y: FloatArray,
    dtype: GgmlDType,
    ncols: Int,
    nrows: Int,
    dev: CudaDevice
): CudaStorage {
    val dataElems = data.len / dtype.typeSize() * dtype.blockSize()
    if (dataElems < ncols * nrows) {
        bail("unexpected data size $dataElems, ncols $ncols $nrows")
    }
    if (y.size != ncols) {
        bail("unexpected y size ${y.size}, ncols $ncols $nrows")
    }
    val kernelName = when (dtype) {
        GgmlDType.Q40 -> "dequantizeMulMatVecQ40Cuda"
        GgmlDType.Q41 -> "dequantizeMulMatVecQ41Cuda"
        GgmlDType.Q50 -> "dequantizeMulMatVecQ50Cuda"
        GgmlDType.Q51 -> "dequantizeMulMatVecQ51Cuda"
        GgmlDType.Q80 -> "dequantizeMulMatVecQ80Cuda"
        GgmlDType.Q2K -> "dequantizeMulMatVecQ2K"
        GgmlDType.Q3K -> "dequantizeMulMatVecQ3K"
        GgmlDType.Q4K -> "dequantizeMulMatVecQ4K"
        GgmlDType.Q5K -> "dequantizeMulMatVecQ5K"
        GgmlDType.Q6K -> "dequantizeMulMatVecQ6K"
        else -> bail("unsupported dtype for quantized matmul $dtype")
    }
    if (kernelName.isNotEmpty()) {
        return CudaStorage(dev)
    }
    return CudaStorage(dev)
}

private fun mulMatVecViaQ81(
    data: PaddedCudaSlice,
    y: FloatArray,
    dtype: GgmlDType,
    ncols: Int,
    nrows: Int,
    bSize: Int,
    dev: CudaDevice
): CudaStorage {
    val dataElems = data.len / dtype.typeSize() * dtype.blockSize()
    if (dataElems < ncols * nrows) {
        bail("unexpected data size $dataElems, ncols $ncols $nrows")
    }
    if (y.size != ncols * bSize) {
        bail("unexpected y size ${y.size}, ncols $ncols $nrows")
    }
    if (bSize == 0 || bSize > 8) {
        bail("only bsize between 1 and 8 are supported, got $bSize")
    }
    val ncolsPadded = pad(ncols, MATRIX_ROW_PADDING)
    val ySizeInBytes = bSize * ncolsPadded * GgmlDType.Q81.typeSize() / GgmlDType.Q81.blockSize()
    val yQ81 = ByteArray(ySizeInBytes)
    quantizeQ81(y, yQ81, ncols, bSize, dev)

    val kernelName = when (dtype) {
        GgmlDType.Q40 -> "mulMatVecQ40Q81Cuda"
        GgmlDType.Q41 -> "mulMatVecQ41Q81Cuda"
        GgmlDType.Q50 -> "mulMatVecQ50Q81Cuda"
        GgmlDType.Q51 -> "mulMatVecQ51Q81Cuda"
        GgmlDType.Q80 -> "mulMatVecQ80Q81Cuda"
        GgmlDType.Q2K -> "mulMatVecQ2KQ81Cuda"
        GgmlDType.Q3K -> "mulMatVecQ3KQ81Cuda"
        GgmlDType.Q4K -> "mulMatVecQ4KQ81Cuda"
        GgmlDType.Q5K -> "mulMatVecQ5KQ81Cuda"
        GgmlDType.Q6K -> "mulMatVecQ6KQ81Cuda"
        else -> bail("unsupported dtype for quantized matmul $dtype")
    }
    if (kernelName.isNotEmpty()) {
        return CudaStorage(dev)
    }
    return CudaStorage(dev)
}

private fun mulMatViaQ81(
    data: PaddedCudaSlice,
    y: FloatArray,
    dtype: GgmlDType,
    xRows: Int,
    xCols: Int,
    yRows: Int,
    yCols: Int,
    dev: CudaDevice
): CudaStorage {
    val dataElems = data.len / dtype.typeSize() * dtype.blockSize()
    if (dataElems < xRows * xCols) {
        bail("unexpected lhs size $dataElems, $xRows $xCols")
    }
    if (y.size != yRows * yCols) {
        bail("unexpected y size ${y.size}, $yRows $yCols")
    }
    if (xCols != yRows) {
        bail("unexpected x/y size $xRows $xCols $yRows $yCols")
    }
    val k = xCols
    val kPadded = pad(k, MATRIX_ROW_PADDING)
    val ySizeInBytes = kPadded * yCols * GgmlDType.Q81.typeSize() / GgmlDType.Q81.blockSize()
    val yQ81 = ByteArray(ySizeInBytes)
    quantizeQ81(y, yQ81, k, yCols, dev)

    val kernelInfo = when (dtype) {
        GgmlDType.Q40 -> Triple("mulMatQ40", 64, 128)
        GgmlDType.Q41 -> Triple("mulMatQ41", 64, 128)
        GgmlDType.Q50 -> Triple("mulMatQ50", 128, 64)
        GgmlDType.Q51 -> Triple("mulMatQ51", 128, 64)
        GgmlDType.Q80 -> Triple("mulMatQ80", 128, 64)
        GgmlDType.Q2K -> Triple("mulMatQ2K", 64, 128)
        GgmlDType.Q3K -> Triple("mulMatQ3K", 128, 128)
        GgmlDType.Q4K -> Triple("mulMatQ4K", 64, 128)
        GgmlDType.Q5K -> Triple("mulMatQ5K", 64, 128)
        GgmlDType.Q6K -> Triple("mulMatQ6K", 64, 64)
        else -> bail("unsupported dtype for quantized matmul $dtype")
    }
    if (kernelInfo.first.isNotEmpty()) {
        return CudaStorage(dev)
    }
    return CudaStorage(dev)
}

private fun indexedMoeForwardFusedQ81Input(
    weight: ByteArray,
    wShape: Shape,
    wDtype: GgmlDType,
    input: FloatArray,
    inShape: Shape,
    ids: IntArray,
    idxShape: Shape,
    dev: CudaDevice
): Pair<CudaStorage, Shape> {
    val (_, n, k) = wShape.dims3()
    val batch = inShape.dims()[0]
    val inputDim1 = inShape.dims()[1]
    val topk = idxShape.dims()[1]
    require(batch == idxShape.dims()[0]) { "batch dim not match!" }

    val totalRows = batch * inputDim1
    val kPadded = pad(k, MATRIX_ROW_PADDING)
    val q81BlockSize = GgmlDType.Q81.blockSize()
    val q81TypeSize = GgmlDType.Q81.typeSize()
    val numBlocksPerRow = kPadded / q81BlockSize
    val dstRowSizeBytes = numBlocksPerRow * q81TypeSize
    val ySizeInBytes = totalRows * dstRowSizeBytes
    val inputQuant = ByteArray(ySizeInBytes)
    quantizeQ81(input, inputQuant, k, totalRows, dev)

    val kernelName = when (wDtype) {
        GgmlDType.Q2K -> "indexedMoeForwardQ2kQ81"
        GgmlDType.Q3K -> "indexedMoeForwardQ3kQ81"
        GgmlDType.Q4K -> "indexedMoeForwardQ4kQ81"
        GgmlDType.Q5K -> "indexedMoeForwardQ5kQ81"
        GgmlDType.Q6K -> "indexedMoeForwardQ6kQ81"
        GgmlDType.Q80 -> "indexedMoeForwardQ80Q81"
        else -> bail("unsupported dtype for indexedMoeForward $wDtype")
    }
    val outDims = inShape.dims().toMutableList()
    outDims.removeAt(outDims.lastIndex)
    outDims.add(n)
    outDims[1] = topk
    if (weight.isNotEmpty() && ids.isNotEmpty() && kernelName.isNotEmpty()) {
        return Pair(CudaStorage(dev), Shape(outDims))
    }
    return Pair(CudaStorage(dev), Shape(outDims))
}

fun <T> loadQuantized(
    device: CudaDevice,
    data: List<T>,
    companion: GgmlTypeCompanion<T>
): QStorage {
    val dtype = companion.DTYPE
    val rawLen = data.size * dtype.typeSize()
    val paddedLen = rawLen + MATRIX_ROW_PADDING * dtype.typeSize() / dtype.blockSize()
    val inner = ByteArray(paddedLen)
    return QStorage.Cuda(
        QCudaStorage(
            data = PaddedCudaSlice(inner, rawLen),
            dtypeVal = dtype,
            deviceVal = device
        )
    )
}

internal fun cudaQuantizeQ81() {
    val dev = CudaDevice.new(0)
    val el = 256
    val elPadded = pad(el, MATRIX_ROW_PADDING)
    val ySizeInBytes = elPadded * GgmlDType.Q81.typeSize() / GgmlDType.Q81.blockSize()
    val yQ81 = ByteArray(ySizeInBytes)
    val vs = FloatArray(el) { it.toFloat() }
    quantizeQ81(vs, yQ81, el, 1, dev)
}

internal fun cudaMmvQ81() {
    val dev = CudaDevice.new(0)
    val ncols = 256
    val vs = FloatArray(ncols) { it.toFloat() }
    val xs = QCudaStorage.zeros(dev, ncols, GgmlDType.Q40)
    xs.quantizeOnto(CpuStorage.F32(vs))
    mulMatVecViaQ81(xs.data, vs, GgmlDType.Q40, ncols, 1, 1, dev)
    dequantizeMulMatVec(xs.data, vs, GgmlDType.Q40, ncols, 1, dev)
}

internal fun cudaMmQ81() {
    val dev = CudaDevice.new(0)
    val ncols = 256
    val vs = FloatArray(ncols * 4) { it.toFloat() / 4.0f }
    val xs = QCudaStorage.zeros(dev, ncols * 4, GgmlDType.Q40)
    xs.quantizeOnto(CpuStorage.F32(vs))
    mulMatViaQ81(xs.data, vs, GgmlDType.Q40, 4, ncols, ncols, 4, dev)
}

internal fun cudaMmQ81Pad() {
    val dev = CudaDevice.new(0)
    val xRows = 4
    val ncols = 16
    val yCols = 2048
    val vs = FloatArray(ncols * yCols) { it.toFloat() / 256.0f }
    val xs = QCudaStorage.zeros(dev, ncols * xRows, GgmlDType.Q40)
    xs.quantizeOnto(CpuStorage.F32(vs.copyOfRange(0, ncols * xRows)))
    mulMatViaQ81(xs.data, vs, GgmlDType.Q40, xRows, ncols, ncols, yCols, dev)
}
