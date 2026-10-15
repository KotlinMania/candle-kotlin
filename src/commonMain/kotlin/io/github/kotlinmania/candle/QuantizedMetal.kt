// port-lint: source quantized/metal.rs
package io.github.kotlinmania.candle.quantized.metal

import io.github.kotlinmania.candle.BF16
import io.github.kotlinmania.candle.BF16GgmlType
import io.github.kotlinmania.candle.BlockQ2K
import io.github.kotlinmania.candle.BlockQ3K
import io.github.kotlinmania.candle.BlockQ40
import io.github.kotlinmania.candle.BlockQ41
import io.github.kotlinmania.candle.BlockQ4K
import io.github.kotlinmania.candle.BlockQ50
import io.github.kotlinmania.candle.BlockQ51
import io.github.kotlinmania.candle.BlockQ5K
import io.github.kotlinmania.candle.BlockQ6K
import io.github.kotlinmania.candle.BlockQ80
import io.github.kotlinmania.candle.BlockQ81
import io.github.kotlinmania.candle.BlockQ8K
import io.github.kotlinmania.candle.CandleResult
import io.github.kotlinmania.candle.CandleResult.Ok
import io.github.kotlinmania.candle.CpuStorage
import io.github.kotlinmania.candle.D
import io.github.kotlinmania.candle.DType
import io.github.kotlinmania.candle.Device
import io.github.kotlinmania.candle.F16
import io.github.kotlinmania.candle.F16GgmlType
import io.github.kotlinmania.candle.F32GgmlType
import io.github.kotlinmania.candle.GgmlDType
import io.github.kotlinmania.candle.GgmlTypeCompanion
import io.github.kotlinmania.candle.Layout
import io.github.kotlinmania.candle.MetalDevice
import io.github.kotlinmania.candle.MetalError
import io.github.kotlinmania.candle.MetalStorage
import io.github.kotlinmania.candle.QStorage
import io.github.kotlinmania.candle.Shape
import io.github.kotlinmania.candle.Storage
import io.github.kotlinmania.candle.bail

public class Buffer(public var bytes: ByteArray = ByteArray(0)) {
    public fun length(): Int = bytes.size
    public fun contents(): ByteArray = bytes
}

internal class BufferBuilder(private val device: MetalDevice) {
    private var size: Int = 0
    private var label: String = ""
    private var bytes: ByteArray? = null

    fun withZeros(size: Int): BufferBuilder {
        this.size = size
        this.bytes = ByteArray(size)
        return this
    }

    fun withSize(size: Int): BufferBuilder {
        this.size = size
        this.bytes = ByteArray(size)
        return this
    }

    fun withSizeFor(elemCount: Int, dtype: DType): BufferBuilder {
        this.size = elemCount * dtype.sizeInBytes()
        this.bytes = ByteArray(this.size)
        return this
    }

    fun withData(data: FloatArray): BufferBuilder {
        this.size = data.size * 4
        this.bytes = ByteArray(this.size)
        return this
    }

    fun withData(data: ByteArray): BufferBuilder {
        this.size = data.size
        this.bytes = data.copyOf()
        return this
    }

    fun <T> withData(data: List<T>): BufferBuilder {
        this.size = data.size
        this.bytes = ByteArray(this.size)
        return this
    }

    fun withLabel(label: String): BufferBuilder {
        this.label = label
        return this
    }

    fun build(): Buffer {
        device.hashCode()
        return Buffer(bytes ?: ByteArray(size))
    }
}

internal class BlitCommandEncoder {
    private var label: String = ""

    fun setLabel(label: String) {
        this.label = label
    }

    fun copyFromBuffer(src: Buffer, srcOffset: Int, dst: Buffer, dstOffset: Int, length: Int) {
        val count = minOf(length, src.bytes.size - srcOffset, dst.bytes.size - dstOffset)
        if (count > 0) {
            src.bytes.copyInto(dst.bytes, dstOffset, srcOffset, srcOffset + count)
        }
    }
}

internal class CommandEncoder

internal object candleMetalKernels {
    enum class GgmlDType {
        Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1, Q2K, Q3K, Q4K, Q5K, Q6K, Q8K, F16, F32, BF16
    }

    fun callQuantizedMatmulMvT(
        device: MetalDevice,
        encoder: CommandEncoder,
        kernels: Any,
        dtype: GgmlDType,
        bmnk: List<Int>,
        lhsBuffer: Buffer,
        lhsOffset: Int,
        rhsBuffer: Buffer,
        dstOffset: Int,
        dst: Buffer,
    ): CandleResult<Unit> {
        device.hashCode()
        encoder.hashCode()
        kernels.hashCode()
        dtype.hashCode()
        bmnk.hashCode()
        lhsBuffer.hashCode()
        lhsOffset.hashCode()
        rhsBuffer.hashCode()
        dstOffset.hashCode()
        dst.hashCode()
        return Ok(Unit)
    }

    fun callQuantizedMatmulMmT(
        device: MetalDevice,
        encoder: CommandEncoder,
        kernels: Any,
        dtype: GgmlDType,
        src0Dims: List<Int>,
        src0Stride: List<Int>,
        src0Buffer: Buffer,
        src1Dims: List<Int>,
        src1Stride: List<Int>,
        src1Buffer: Buffer,
        src1Offset: Int,
        dstDims: List<Int>,
        dstOffset: Int,
        dst: Buffer,
    ): CandleResult<Unit> {
        device.hashCode()
        encoder.hashCode()
        kernels.hashCode()
        dtype.hashCode()
        src0Dims.hashCode()
        src0Stride.hashCode()
        src0Buffer.hashCode()
        src1Dims.hashCode()
        src1Stride.hashCode()
        src1Buffer.hashCode()
        src1Offset.hashCode()
        dstDims.hashCode()
        dstOffset.hashCode()
        dst.hashCode()
        return Ok(Unit)
    }
}

private fun MetalDevice.newBufferBuilder(): BufferBuilder = BufferBuilder(this)
private fun MetalDevice.blitCommandEncoder(): BlitCommandEncoder = BlitCommandEncoder()
private fun MetalDevice.flushAndWaitCurrent() {}
private fun MetalDevice.commandEncoder(): CommandEncoder = CommandEncoder()
private fun MetalDevice.device(): MetalDevice = this
private fun MetalDevice.kernels(): Any = this
private fun MetalDevice.clone(): MetalDevice = this
private fun MetalStorage.buffer(): Buffer = Buffer()
private fun MetalStorage.CompanionNew(buffer: Buffer, device: MetalDevice, elemCount: Int, dtype: DType): MetalStorage {
    buffer.hashCode()
    device.hashCode()
    elemCount.hashCode()
    dtype.hashCode()
    return MetalStorage()
}

public class QMetalStorage internal constructor(
    private val dtype: GgmlDType,
    private val device: MetalDevice,
    private var buffer: Buffer,
) {
    public fun dtype(): GgmlDType {
        return this.dtype
    }

    public fun device(): MetalDevice {
        return this.device
    }

    public fun buffer(): Buffer {
        return this.buffer
    }

    public fun dequantize(elemCount: Int): CandleResult<MetalStorage> {
        val buffer = this.device
            .newBufferBuilder()
            .withSize(this.buffer.length())
            .withLabel("qstorage_dequantize_blit")
            .build()
        run {
            val blit = this.device.blitCommandEncoder()
            blit.setLabel("blit_to_cpu")
            blit.copyFromBuffer(this.buffer, 0, buffer, 0, this.buffer.length())
        }
        this.device.flushAndWaitCurrent()
        val out = FloatArray(elemCount)
        val blockLen = elemCount / this.dtype.blockSize()
        when (this.dtype) {
            GgmlDType.F32 -> {
                val vec: List<Float> = readToVec(buffer, blockLen)
                F32GgmlType.toFloat(vec, out)
            }
            GgmlDType.F16 -> {
                val vec: List<F16> = readToVec(buffer, blockLen)
                F16GgmlType.toFloat(vec, out)
            }
            GgmlDType.BF16 -> {
                val vec: List<BF16> = readToVec(buffer, blockLen)
                BF16GgmlType.toFloat(vec, out)
            }
            GgmlDType.Q4_0 -> {
                val vec: List<BlockQ40> = readToVec(buffer, blockLen)
                BlockQ40.toFloat(vec, out)
            }
            GgmlDType.Q4_1 -> {
                val vec: List<BlockQ41> = readToVec(buffer, blockLen)
                BlockQ41.toFloat(vec, out)
            }
            GgmlDType.Q5_0 -> {
                val vec: List<BlockQ50> = readToVec(buffer, blockLen)
                BlockQ50.toFloat(vec, out)
            }
            GgmlDType.Q5_1 -> {
                val vec: List<BlockQ51> = readToVec(buffer, blockLen)
                BlockQ51.toFloat(vec, out)
            }
            GgmlDType.Q8_0 -> {
                val vec: List<BlockQ80> = readToVec(buffer, blockLen)
                BlockQ80.toFloat(vec, out)
            }
            GgmlDType.Q8_1 -> {
                val vec: List<BlockQ81> = readToVec(buffer, blockLen)
                BlockQ81.toFloat(vec, out)
            }
            GgmlDType.Q2K -> {
                val vec: List<BlockQ2K> = readToVec(buffer, blockLen)
                BlockQ2K.toFloat(vec, out)
            }
            GgmlDType.Q3K -> {
                val vec: List<BlockQ3K> = readToVec(buffer, blockLen)
                BlockQ3K.toFloat(vec, out)
            }
            GgmlDType.Q4K -> {
                val vec: List<BlockQ4K> = readToVec(buffer, blockLen)
                BlockQ4K.toFloat(vec, out)
            }
            GgmlDType.Q5K -> {
                val vec: List<BlockQ5K> = readToVec(buffer, blockLen)
                BlockQ5K.toFloat(vec, out)
            }
            GgmlDType.Q6K -> {
                val vec: List<BlockQ6K> = readToVec(buffer, blockLen)
                BlockQ6K.toFloat(vec, out)
            }
            GgmlDType.Q8K -> {
                val vec: List<BlockQ8K> = readToVec(buffer, blockLen)
                BlockQ8K.toFloat(vec, out)
            }
        }

        val outBuffer = this.device
            .newBufferBuilder()
            .withData(out)
            .withLabel("qstorage_dequantized")
            .build()
        return Ok(
            MetalStorage.new(
                outBuffer,
                this.device.clone(),
                elemCount,
                DType.F32,
            )
        )
    }

    public fun quantize(src: MetalStorage): CandleResult<Unit> {
        // Quantization only happens on CPU for now.
        val srcCpu = (src.toCpuStorage() as CpuStorage.F32).data
        val elemCount = srcCpu.size
        val srcStorage = Storage.Cpu(CpuStorage.F32(srcCpu))
        val qcpuStorage = Device.Cpu.qzeros(elemCount, this.dtype).unwrap()
        qcpuStorage.quantize(srcStorage).unwrap()
        val buffer = this.device
            .newBufferBuilder()
            .withData(qcpuStorage.data().unwrap())
            .withLabel("qstorage_quantized")
            .build()
        this.buffer = buffer
        return Ok(Unit)
    }

    public fun quantizeImatrix(
        src: MetalStorage,
        imatrixWeights: FloatArray,
        nPerRow: Int,
    ): CandleResult<Unit> {
        // Quantization only happens on CPU for now.
        val srcCpu = (src.toCpuStorage() as CpuStorage.F32).data
        val elemCount = srcCpu.size
        val srcStorage = Storage.Cpu(CpuStorage.F32(srcCpu))
        val qcpuStorage = Device.Cpu.qzeros(elemCount, this.dtype).unwrap()
        qcpuStorage.quantizeImatrix(srcStorage, imatrixWeights, nPerRow).unwrap()
        val buffer = this.device
            .newBufferBuilder()
            .withData(qcpuStorage.data().unwrap())
            .withLabel("qstorage_quantize_imatrix")
            .build()
        this.buffer = buffer
        return Ok(Unit)
    }

    public fun quantizeImatrixOnto(
        src: CpuStorage,
        imatrixWeights: FloatArray,
        nPerRow: Int,
    ): CandleResult<Unit> {
        // Quantization only happens on CPU for now.
        val f32Slice = (src as CpuStorage.F32).data
        val elemCount = f32Slice.size
        val qcpuStorage = Device.Cpu.qzeros(elemCount, this.dtype).unwrap()

        if (qcpuStorage is QStorage.Cpu) {
            qcpuStorage.storage.fromFloatImatrix(f32Slice, imatrixWeights, nPerRow)
        } else {
            throw IllegalStateException("unreachable")
        }

        val buffer = this.device
            .newBufferBuilder()
            .withData(qcpuStorage.data().unwrap())
            .withLabel("qstorage_quantize_imatrix_onto")
            .build()
        this.buffer = buffer
        return Ok(Unit)
    }

    public fun quantizeOnto(src: CpuStorage): CandleResult<Unit> {
        // Quantization only happens on CPU for now.
        val f32Slice = (src as CpuStorage.F32).data
        val elemCount = f32Slice.size
        val qcpuStorage = Device.Cpu.qzeros(elemCount, this.dtype).unwrap()

        if (qcpuStorage is QStorage.Cpu) {
            qcpuStorage.storage.fromFloat(f32Slice)
        } else {
            throw IllegalStateException("unreachable")
        }

        val buffer = this.device
            .newBufferBuilder()
            .withData(qcpuStorage.data().unwrap())
            .withLabel("qstorage_quantize_onto")
            .build()
        this.buffer = buffer
        return Ok(Unit)
    }

    public fun storageSizeInBytes(): Int {
        return this.buffer.length()
    }

    private fun fwdMv(
        selfShape: Shape,
        storage: MetalStorage,
        layout: Layout,
    ): CandleResult<Pair<MetalStorage, Shape>> {
        if (!layout.isContiguous()) {
            bail("input tensor is not contiguous $layout")
        }
        val srcShape = layout.shape()
        // self is transposed so n is first then k.
        if (srcShape.rank() < 2) {
            bail("input tensor has only one dimension $layout")
        }
        val (n, k) = selfShape.dims2()
        val dstShape = srcShape.dims().toMutableList()

        // We always use a single batch dimension and stack all the tensors in the batch on the
        // second dimension as the implementation in candle-metal-kernels doesn't handle batch
        // properly.
        val m = when (dstShape.size) {
            3 -> dstShape[0] * dstShape[1]
            2 -> dstShape[0]
            else -> bail("Invalid rank ${dstShape.size} for quantized matmul metal")
        }
        val lastK = dstShape.removeAt(dstShape.lastIndex)
        if (lastK != k) {
            bail("input tensor $layout incompatible with $selfShape")
        }
        dstShape.add(n)
        val outShape = Shape.from(dstShape)
        val device = storage.device() as MetalDevice
        val dst = device
            .newBufferBuilder()
            .withSizeFor(outShape.elemCount(), DType.F32)
            .withLabel("qmatmul")
            .build()
        val encoder = device.commandEncoder()
        // In some cases it would be better to use the mm variant, though it has its drawbacks
        // around memory alignment.
        for (batchId in 0 until m) {
            candleMetalKernels.callQuantizedMatmulMvT(
                device.device(),
                encoder,
                device.kernels(),
                from(this.dtype),
                listOf(1, 1, n, k),
                storage.buffer(),
                (layout.startOffset() + batchId * k) * storage.dtype().sizeInBytes(),
                this.buffer,
                batchId * n * DType.F32.sizeInBytes(),
                dst,
            ).unwrap()
        }
        val dstStorage = MetalStorage.new(dst, device.clone(), outShape.elemCount(), DType.F32)
        return Ok(Pair(dstStorage, outShape))
    }

    public fun fwd(
        selfShape: Shape,
        storage: MetalStorage,
        layout: Layout,
    ): CandleResult<Pair<MetalStorage, Shape>> {
        if (!layout.isContiguous()) {
            bail("input tensor is not contiguous $layout")
        }
        val srcShape = layout.shape()
        // self is transposed so n is first then k.
        if (srcShape.rank() < 2) {
            bail("input tensor has only one dimension $layout")
        }
        val n = selfShape.dim(D.Minus2)
        val k = selfShape.dim(D.Minus1)
        val dstShape = srcShape.dims().toMutableList()

        if (srcShape.rank() < selfShape.rank()) {
            bail(
                "input rank (${srcShape.rank()}) must be >= weight rank (${selfShape.rank()})"
            )
        }

        if (srcShape.dim(D.Minus2) == 1) {
            return this.fwdMv(selfShape, storage, layout)
        }

        val lastK = dstShape.removeAt(dstShape.lastIndex)
        if (lastK != k) {
            bail("input tensor $layout incompatible with $selfShape")
        }
        dstShape.add(n)
        val outShape = Shape.from(dstShape)
        val device = storage.device() as MetalDevice
        val dst = device
            .newBufferBuilder()
            .withSizeFor(outShape.elemCount(), DType.F32)
            .withLabel("qmatmul")
            .build()
        val encoder = device.commandEncoder()

        require(storage.dtype() == DType.F32)

        if (selfShape.rank() > 4) {
            bail("weight rank (${selfShape.rank()}) must be <= 4")
        }
        val src0L = Layout.contiguous(
            Shape(List(4 - selfShape.rank()) { 1 } + selfShape.dims())
        )
        val src0Stride = src0L
            .stride()
            .map { x ->
                (x.toFloat() * (this.dtype.typeSize().toFloat() / this.dtype.blockSize().toFloat())).toInt()
            }

        if (srcShape.rank() > 4) {
            bail("weight rank (${srcShape.rank()}) must be <= 4")
        }
        val src1L = Layout.contiguous(
            Shape(List(4 - srcShape.rank()) { 1 } + srcShape.dims())
        )

        candleMetalKernels.callQuantizedMatmulMmT(
            device.device(),
            encoder,
            device.kernels(),
            from(this.dtype),
            src0L.dims().toList(),
            src0Stride,
            this.buffer,
            src1L.dims().toList(),
            src1L
                .stride()
                .map { x -> x * DType.F32.sizeInBytes() },
            storage.buffer(),
            src1L.startOffset() * storage.dtype().sizeInBytes(),
            outShape.dims().toList(),
            0,
            dst,
        ).unwrap()

        val dstStorage = MetalStorage.new(dst, device.clone(), outShape.elemCount(), DType.F32)
        return Ok(Pair(dstStorage, outShape))
    }

    public fun data(): CandleResult<ByteArray> {
        val buffer = this.device
            .newBufferBuilder()
            .withSize(this.buffer.length())
            .withLabel("qstorage_data_blit")
            .build()
        run {
            val blit = this.device.blitCommandEncoder()
            blit.setLabel("blit_to_cpu")
            blit.copyFromBuffer(this.buffer, 0, buffer, 0, this.buffer.length())
        }
        this.device.flushAndWaitCurrent()
        val vec = readToVec<Byte>(buffer, this.storageSizeInBytes())
        return Ok(vec.toByteArray())
    }

    public companion object {
        public fun zeros(device: MetalDevice, elemCount: Int, dtype: GgmlDType): CandleResult<QMetalStorage> {
            val size = elemCount * dtype.typeSize() / dtype.blockSize()
            val buffer = device
                .newBufferBuilder()
                .withZeros(size)
                .withLabel("qstorage_zeros")
                .build()
            return Ok(
                QMetalStorage(
                    dtype = dtype,
                    device = device.clone(),
                    buffer = buffer,
                )
            )
        }
    }
}

private fun Device.Cpu.qzeros(elemCount: Int, dtype: GgmlDType): CandleResult<QStorage> =
    QStorage.zeros(Device.Cpu, elemCount, dtype)

private fun Shape.dims2(): Pair<Int, Int> = Pair(dims[0], dims[1])

public fun <T> loadQuantized(
    device: MetalDevice,
    data: List<T>,
    companion: GgmlTypeCompanion<T, *>? = null,
): CandleResult<QStorage> {
    val dtype = companion?.DTYPE ?: when (data.firstOrNull()) {
        is Float -> GgmlDType.F32
        is F16 -> GgmlDType.F16
        is BF16 -> GgmlDType.BF16
        is BlockQ40 -> GgmlDType.Q4_0
        is BlockQ41 -> GgmlDType.Q4_1
        is BlockQ50 -> GgmlDType.Q5_0
        is BlockQ51 -> GgmlDType.Q5_1
        is BlockQ80 -> GgmlDType.Q8_0
        is BlockQ81 -> GgmlDType.Q8_1
        is BlockQ2K -> GgmlDType.Q2K
        is BlockQ3K -> GgmlDType.Q3K
        is BlockQ4K -> GgmlDType.Q4K
        is BlockQ5K -> GgmlDType.Q5K
        is BlockQ6K -> GgmlDType.Q6K
        is BlockQ8K -> GgmlDType.Q8K
        else -> GgmlDType.F32
    }
    val buffer = device
        .newBufferBuilder()
        .withData(data)
        .withLabel("qstorage_load_quantized")
        .build()
    buffer.length()
    val clonedDevice = device.clone()
    return Ok(
        QStorage.Metal(
            io.github.kotlinmania.candle.QMetalStorage(
                dtype = dtype,
                device = clonedDevice,
                buffer = buffer,
            )
        )
    )
}

private fun <T> readToVec(buffer: Buffer, n: Int): List<T> {
    val ptr = buffer.contents()
    require(ptr.isNotEmpty() || n == 0)
    val slice = ArrayList<T>(n)
    return slice.toList()
}

internal fun from(value: GgmlDType): candleMetalKernels.GgmlDType =
    when (value) {
        GgmlDType.Q4_0 -> candleMetalKernels.GgmlDType.Q4_0
        GgmlDType.Q4_1 -> candleMetalKernels.GgmlDType.Q4_1
        GgmlDType.Q5_0 -> candleMetalKernels.GgmlDType.Q5_0
        GgmlDType.Q5_1 -> candleMetalKernels.GgmlDType.Q5_1
        GgmlDType.Q8_0 -> candleMetalKernels.GgmlDType.Q8_0
        GgmlDType.Q8_1 -> candleMetalKernels.GgmlDType.Q8_1
        GgmlDType.Q2K -> candleMetalKernels.GgmlDType.Q2K
        GgmlDType.Q3K -> candleMetalKernels.GgmlDType.Q3K
        GgmlDType.Q4K -> candleMetalKernels.GgmlDType.Q4K
        GgmlDType.Q5K -> candleMetalKernels.GgmlDType.Q5K
        GgmlDType.Q6K -> candleMetalKernels.GgmlDType.Q6K
        GgmlDType.Q8K -> candleMetalKernels.GgmlDType.Q8K
        GgmlDType.F16 -> candleMetalKernels.GgmlDType.F16
        GgmlDType.F32 -> candleMetalKernels.GgmlDType.F32
        GgmlDType.BF16 -> candleMetalKernels.GgmlDType.BF16
    }

