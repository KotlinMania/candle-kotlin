// port-lint: source quantized/metal.rs
package io.github.kotlinmania.candle.quantized.metal

import io.github.kotlinmania.candle.cpubackend.CpuStorage
import io.github.kotlinmania.candle.device.Device
import io.github.kotlinmania.candle.dtype.DType
import io.github.kotlinmania.candle.dummymetalbackend.MetalDevice
import io.github.kotlinmania.candle.dummymetalbackend.MetalError
import io.github.kotlinmania.candle.dummymetalbackend.MetalStorage
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.layout.Layout
import io.github.kotlinmania.candle.quantized.GgmlDType
import io.github.kotlinmania.candle.quantized.QStorage
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
import io.github.kotlinmania.candle.quantized.kquants.GgmlType
import io.github.kotlinmania.candle.shape.D
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.candle.storage.Storage
import io.github.kotlinmania.candlemetalkernels.callQuantizedGetRows
import io.github.kotlinmania.candlemetalkernels.callQuantizedMatmulMmT
import io.github.kotlinmania.candlemetalkernels.callQuantizedMatmulMvT
import io.github.kotlinmania.candlemetalkernels.metal.Buffer
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.half.f16.toFloat
import io.github.kotlinmania.rustlang.core.mem.constPtrFromRustCast
import io.github.kotlinmania.rustlang.core.result.fromResidual

public data class QMetalStorage(
    public var dtype: GgmlDType = GgmlDType.entries.first(),
    public var device: MetalDevice = MetalDevice(),
    public var buffer: Buffer = Buffer(),
) {
    public fun dequantize(elemCount: Int): Result<MetalStorage> {
        val buffer =
            when (
                val tryResult =
                    this.device
                        .newBufferBuilder()
                        .withSize(this.buffer.length())
                        .withLabel("qstorage_dequantize_blit")
                        .build()
            ) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        run {
            val blit =
                when (val tryResult = this.device.blitCommandEncoder()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            blit.setLabel("blit_to_cpu")
            blit.copyFromBuffer(this.buffer, 0, buffer, 0, this.buffer.length())
        }
        when (val tryResult = this.device.flushAndWaitCurrent()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        val out = mutableListOf(0.0, elemCount)
        val blockLen = elemCount / this.dtype.blockSize()
        when (this.dtype) {
            GgmlDType.f32 -> {
                val vec: List<Float> = readToVec(buffer, blockLen)
                Float.toFloat(vec, out)
            }
            GgmlDType.f16 -> {
                val vec1: List<F16> = readToVec(buffer, blockLen)
                toFloat(vec1, out)
            }
            GgmlDType.bf16 -> {
                val vec2: List<Bf16> = readToVec(buffer, blockLen)
                io.github.kotlinmania.half.bf16
                    .toFloat(vec2, out)
            }
            GgmlDType.q40 -> {
                val vec3: List<BlockQ40> = readToVec(buffer, blockLen)
                BlockQ40.toFloat(vec3, out)
            }
            GgmlDType.q41 -> {
                val vec4: List<BlockQ41> = readToVec(buffer, blockLen)
                BlockQ41.toFloat(vec4, out)
            }
            GgmlDType.q50 -> {
                val vec5: List<BlockQ50> = readToVec(buffer, blockLen)
                BlockQ50.toFloat(vec5, out)
            }
            GgmlDType.q51 -> {
                val vec6: List<BlockQ51> = readToVec(buffer, blockLen)
                BlockQ51.toFloat(vec6, out)
            }
            GgmlDType.q80 -> {
                val vec7: List<BlockQ80> = readToVec(buffer, blockLen)
                BlockQ80.toFloat(vec7, out)
            }
            GgmlDType.q81 -> {
                val vec8: List<BlockQ81> = readToVec(buffer, blockLen)
                BlockQ81.toFloat(vec8, out)
            }
            GgmlDType.q2k -> {
                val vec9: List<BlockQ2K> = readToVec(buffer, blockLen)
                BlockQ2K.toFloat(vec9, out)
            }
            GgmlDType.q3k -> {
                val vec10: List<BlockQ3K> = readToVec(buffer, blockLen)
                BlockQ3K.toFloat(vec10, out)
            }
            GgmlDType.q4k -> {
                val vec11: List<BlockQ4K> = readToVec(buffer, blockLen)
                BlockQ4K.toFloat(vec11, out)
            }
            GgmlDType.q5k -> {
                val vec12: List<BlockQ5K> = readToVec(buffer, blockLen)
                BlockQ5K.toFloat(vec12, out)
            }
            GgmlDType.q6k -> {
                val vec13: List<BlockQ6K> = readToVec(buffer, blockLen)
                BlockQ6K.toFloat(vec13, out)
            }
            GgmlDType.q8k -> {
                val vec14: List<BlockQ8K> = readToVec(buffer, blockLen)
                BlockQ8K.toFloat(vec14, out)
            }
        }
        val buffer1 =
            when (
                val tryResult =
                    this.device
                        .newBufferBuilder()
                        .withData(out)
                        .withLabel("qstorage_dequantized")
                        .build()
            ) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        return Ok(MetalStorage.new(buffer1, this.device.clone(), elemCount, DType.F32))
    }

    public fun quantize(src: MetalStorage): Result<Unit> {
        // Quantization only happens on CPU for now.
        val src1 =
            when (val tryResult = src.toCpu<Float>()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val elemCount = src1.len()
        val src2 = Storage.Cpu(CpuStorage.F32(src1))
        val qcpuStorage =
            when (val tryResult = Device.Cpu.qzeros(elemCount, this.dtype)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        when (val tryResult = qcpuStorage.quantize(src2)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        val buffer =
            when (
                val tryResult =
                    this.device
                        .newBufferBuilder()
                        .withData(
                            when (val tryResult = qcpuStorage.data()) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        ).withLabel("qstorage_quantized")
                        .build()
            ) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        this.buffer = buffer
        return Ok(Unit)
    }

    public fun quantizeImatrix(src: MetalStorage, imatrixWeights: List<Float>, nPerRow: Int): Result<Unit> {
        // Quantization only happens on CPU for now.
        val src1 =
            when (val tryResult = src.toCpu<Float>()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val elemCount = src1.len()
        val src2 = Storage.Cpu(CpuStorage.F32(src1))
        val qcpuStorage =
            when (val tryResult = Device.Cpu.qzeros(elemCount, this.dtype)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        when (val tryResult = qcpuStorage.quantizeImatrix(src2, imatrixWeights, nPerRow)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        val buffer =
            when (
                val tryResult =
                    this.device
                        .newBufferBuilder()
                        .withData(
                            when (val tryResult = qcpuStorage.data()) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        ).withLabel("qstorage_quantize_imatrix")
                        .build()
            ) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        this.buffer = buffer
        return Ok(Unit)
    }

    public fun quantizeImatrixOnto(src: CpuStorage, imatrixWeights: List<Float>, nPerRow: Int): Result<Unit> {
        // Quantization only happens on CPU for now.
        val elemCount =
            when (val tryResult = src.asSlice<Float>()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }.len()
        val qcpuStorage =
            when (val tryResult = Device.Cpu.qzeros(elemCount, this.dtype)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        if (qcpuStorage is QStorage.Cpu) {
            val (storage) = qcpuStorage
            storage.fromFloatImatrix(
                when (val tryResult = src.asSlice<Float>()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                },
                imatrixWeights,
                nPerRow,
            )
        } else {
            error("unreachable")
        }
        val buffer =
            when (
                val tryResult =
                    this.device
                        .newBufferBuilder()
                        .withData(
                            when (val tryResult = qcpuStorage.data()) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        ).withLabel("qstorage_quantize_imatrix_onto")
                        .build()
            ) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        this.buffer = buffer
        return Ok(Unit)
    }

    public fun quantizeOnto(src: CpuStorage): Result<Unit> {
        // Quantization only happens on CPU for now.
        val elemCount =
            when (val tryResult = src.asSlice<Float>()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }.len()
        val qcpuStorage =
            when (val tryResult = Device.Cpu.qzeros(elemCount, this.dtype)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        if (qcpuStorage is QStorage.Cpu) {
            val (storage) = qcpuStorage
            storage.fromFloat(
                when (val tryResult = src.asSlice<Float>()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                },
            )
        } else {
            error("unreachable")
        }
        val buffer =
            when (
                val tryResult =
                    this.device
                        .newBufferBuilder()
                        .withData(
                            when (val tryResult = qcpuStorage.data()) {
                                is Result.Ok -> tryResult.value
                                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                            },
                        ).withLabel("qstorage_quantize_onto")
                        .build()
            ) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        this.buffer = buffer
        return Ok(Unit)
    }

    public fun storageSizeInBytes(): Int = this.buffer.length()

    public fun embedding(rows: Int, hidden: Int, ids: MetalStorage, idsL: Layout): Result<MetalStorage> {
        if (ids.dtype() != DType.U32) {
            bail("quantized embedding expects u32 ids, got {:?}", ids.dtype())
        }
        if (!idsL.isContiguous()) {
            bail("quantized embedding requires contiguous ids")
        }
        if (!hidden.isMultipleOf(this.dtype.blockSize())) {
            bail("quantized embedding hidden size {hidden} is not divisible by block size {}", this.dtype.blockSize())
        }
        val expectedSize = rows * hidden * this.dtype.typeSize() / this.dtype.blockSize()
        if (this.storageSizeInBytes() != expectedSize) {
            bail("quantized tensor has {} bytes, expected {expected_size}", this.storageSizeInBytes())
        }
        val idsLen = idsL.shape().elemCount()
        val device = this.device.clone()
        val dst =
            when (
                val tryResult =
                    device
                        .newBufferBuilder()
                        .withSizeFor(idsLen * hidden, DType.F32)
                        .withLabel("qembedding")
                        .build()
            ) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val encoder =
            when (val tryResult = device.commandEncoder()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        when (val tryResult = callQuantizedGetRows(device.device(), encoder, device.kernels(), this.dtype.into(), hidden, hidden * this.dtype.typeSize() / this.dtype.blockSize(), idsLen, this.buffer, ids.buffer(), idsL.startOffset() * DType.U32.sizeInBytes(), dst).mapErr(MetalError::from)) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(MetalStorage.new(dst, device.clone(), idsLen * hidden, DType.F32))
    }

    public fun fwdMv(selfShape: Shape, storage: MetalStorage, layout: Layout): Result<Pair<MetalStorage, Shape>> {
        if (!layout.isContiguous()) {
            bail("input tensor is not contiguous {layout:?}")
        }
        val srcShape = layout.shape()
        // self is transposed so n is first then k.
        if (srcShape.rank() < 2) {
            bail("input tensor has only one dimension {layout:?}")
        }
        val (n, k) =
            when (val tryResult = selfShape.dims2()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val dstShape = srcShape.dims().toVec()
        // We always use a single batch dimension and stack all the tensors in the batch on the
        // second dimension as the implementation in candle-metal-kernels doesn't handle batch
        // properly.
        val m =
            when (val matchVal = dstShape.len()) {
                3 -> dstShape[0] * dstShape[1]
                2 -> dstShape[0]
                else -> {
                    val n1 = matchVal
                    bail("Invalid rank {n} for quantized matmul metal")
                }
            }
        val lastK = dstShape.pop()!!
        if (lastK != k) {
            bail("input tensor {layout:?} incompatible with {:?}", selfShape)
        }
        dstShape.push(n)
        val dstShape1 = Shape.from(dstShape)
        val device = storage.device().clone()
        val dst =
            when (
                val tryResult =
                    device
                        .newBufferBuilder()
                        .withSizeFor(dstShape1.elemCount(), DType.F32)
                        .withLabel("qmatmul")
                        .build()
            ) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val encoder =
            when (val tryResult = device.commandEncoder()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        // In some cases it would be better to use the mm variant, though it has its drawbacks
        // around memory alignment.
        for (batchId in 0 until m) {
            when (val tryResult = callQuantizedMatmulMvT(device.device(), encoder, device.kernels(), this.dtype.into(), listOf(1, 1, n, k), storage.buffer(), (layout.startOffset() + batchId * k) * storage.dtype().sizeInBytes(), this.buffer, batchId * n * DType.F32.sizeInBytes(), dst).mapErr(MetalError::from)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        }
        val dstStorage = MetalStorage.new(dst, device.clone(), dstShape1.elemCount(), DType.F32)
        return Ok(Pair(dstStorage, dstShape1))
    }

    public fun fwd(selfShape: Shape, storage: MetalStorage, layout: Layout): Result<Pair<MetalStorage, Shape>> {
        if (!layout.isContiguous()) {
            bail("input tensor is not contiguous {layout:?}")
        }
        val srcShape = layout.shape()
        // self is transposed so n is first then k.
        if (srcShape.rank() < 2) {
            bail("input tensor has only one dimension {layout:?}")
        }
        val n =
            when (val tryResult = selfShape.dim(D.Minus2)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val k =
            when (val tryResult = selfShape.dim(D.Minus1)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val dstShape = srcShape.dims().toVec()
        if (srcShape.rank() < selfShape.rank()) {
            bail("input rank ({}) must be >= weight rank ({})", srcShape.rank(), selfShape.rank())
        }
        if (when (val tryResult = srcShape.dim(D.Minus2)) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            } == 1
        ) {
            return this.fwdMv(selfShape, storage, layout)
        }
        val lastK = dstShape.pop()!!
        if (lastK != k) {
            bail("input tensor {layout:?} incompatible with {:?}", selfShape)
        }
        dstShape.push(n)
        val dstShape1 = Shape.from(dstShape)
        val device = storage.device().clone()
        val dst =
            when (
                val tryResult =
                    device
                        .newBufferBuilder()
                        .withSizeFor(dstShape1.elemCount(), DType.F32)
                        .withLabel("qmatmul")
                        .build()
            ) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        val encoder =
            when (val tryResult = device.commandEncoder()) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        check(storage.dtype() == DType.F32)
        if (selfShape.rank() > 4) {
            bail("weight rank ({}) must be <= 4", selfShape.rank())
        }
        val src0L = Layout.contiguous(listOf(mutableListOf(1, 4 - selfShape.rank()), selfShape.dims().toVec()).concat())
        val src0Stride =
            src0L
                .stride()
                .iter()
                .map({ x ->
                    (x.toFloat() * (this.dtype.typeSize().toFloat() / this.dtype.blockSize().toFloat())).toInt()
                })
                .collect<Vec>()
        if (srcShape.rank() > 4) {
            bail("weight rank ({}) must be <= 4", srcShape.rank())
        }
        val src1L = Layout.contiguous(listOf(mutableListOf(1, 4 - srcShape.rank()), srcShape.dims().toVec()).concat())
        when (
            val tryResult =
                callQuantizedMatmulMmT(
                    device.device(),
                    encoder,
                    device.kernels(),
                    this.dtype.into(),
                    src0L.dims(),
                    src0Stride,
                    this.buffer,
                    src1L.dims(),
                    src1L
                        .stride()
                        .iter()
                        .map({ x -> x * DType.F32.sizeInBytes() })
                        .collect<Vec>(),
                    storage.buffer(),
                    src1L.startOffset() * storage.dtype().sizeInBytes(),
                    dstShape1.dims(),
                    0,
                    dst,
                ).mapErr(MetalError::from)
        ) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        val dstStorage = MetalStorage.new(dst, device.clone(), dstShape1.elemCount(), DType.F32)
        return Ok(Pair(dstStorage, dstShape1))
    }

    public fun data(): Result<List<UByte>> {
        val buffer =
            when (
                val tryResult =
                    this.device
                        .newBufferBuilder()
                        .withSize(this.buffer.length())
                        .withLabel("qstorage_data_blit")
                        .build()
            ) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        run {
            val blit =
                when (val tryResult = this.device.blitCommandEncoder()) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            blit.setLabel("blit_to_cpu")
            blit.copyFromBuffer(this.buffer, 0, buffer, 0, this.buffer.length())
        }
        when (val tryResult = this.device.flushAndWaitCurrent()) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        return Ok(readToVec<UByte>(buffer, this.storageSizeInBytes()))
    }

    public companion object {
        public fun zeros(device: MetalDevice, elemCount: Int, dtype: GgmlDType): Result<QMetalStorage> {
            val size = elemCount * dtype.typeSize() / dtype.blockSize()
            val buffer =
                when (
                    val tryResult =
                        device
                            .newBufferBuilder()
                            .withZeros(size)
                            .withLabel("qstorage_zeros")
                            .build()
                ) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            return Ok(QMetalStorage(buffer = buffer, device = device.clone(), dtype = dtype))
        }
    }
}

public fun <T : GgmlType> loadQuantized(device: MetalDevice, data: List<T>): Result<QStorage> {
    val buffer =
        when (
            val tryResult =
                device
                    .newBufferBuilder()
                    .withData(data)
                    .withLabel("qstorage_load_quantized")
                    .build()
        ) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val device1 = device.clone()
    return Ok(QStorage.Metal(QMetalStorage(dtype = T.DTYPE, device = device1, buffer = buffer)))
}

public fun <T> readToVec(buffer: Buffer, n: Int): List<T> {
    val ptr = constPtrFromRustCast<T>(buffer.contents())
    check(!((ptr).rawAddress == 0L))
    val slice = fromRawParts(ptr, n)
    return slice.toVec()
}

public fun fromGgmlDType(value: GgmlDType): io.github.kotlinmania.candlemetalkernels.GgmlDType =
    when (value) {
        GgmlDType.q40 -> GgmlDType.Q4_0
        GgmlDType.q41 -> GgmlDType.Q4_1
        GgmlDType.q50 -> GgmlDType.Q5_0
        GgmlDType.q51 -> GgmlDType.Q5_1
        GgmlDType.q80 -> GgmlDType.Q8_0
        GgmlDType.q81 -> GgmlDType.Q8_1
        GgmlDType.q2k -> GgmlDType.Q2K
        GgmlDType.q3k -> GgmlDType.Q3K
        GgmlDType.q4k -> GgmlDType.Q4K
        GgmlDType.q5k -> GgmlDType.Q5K
        GgmlDType.q6k -> GgmlDType.Q6K
        GgmlDType.q8k -> GgmlDType.Q8K
        GgmlDType.f16 -> GgmlDType.F16
        GgmlDType.f32 -> GgmlDType.F32
        GgmlDType.bf16 -> GgmlDType.BF16
    }
