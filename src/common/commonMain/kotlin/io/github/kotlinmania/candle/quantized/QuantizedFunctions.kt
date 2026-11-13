package io.github.kotlinmania.candle.quantized

import io.github.kotlinmania.candle.cpubackend.CpuStorage
import io.github.kotlinmania.candle.device.Device
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.candle.quantized.cuda.QCudaStorage
import io.github.kotlinmania.candle.quantized.kquants.GgmlType
import io.github.kotlinmania.candle.quantized.kquants.matmul
import io.github.kotlinmania.candle.quantized.kquants.matmulF16
import io.github.kotlinmania.candle.quantized.metal.QMetalStorage
import io.github.kotlinmania.candle.shape.Shape
import io.github.kotlinmania.candle.utils.barrierPool
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.half.bf16.fromF32
import io.github.kotlinmania.rustlang.core.mem.ObjectSlotLayout
import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.mem.TypeLayout
import io.github.kotlinmania.rustlang.core.mem.constPtrFromRustCast
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr

public fun <T> asTSlice(data: List<UByte>, typeLayoutT: TypeLayout<T>): List<T> {
    val size = sizeOf<T>(typeLayoutT = typeLayoutT)
    check(data.size % size == 0)
    val ptr = data.asPtr()
    check((ptr.toInt()) % alignOf<T>(typeLayoutT = typeLayoutT) == 0)
    return fromRawParts(constPtrFromRustCast(ptr), data.size / size)
}

public fun checkShape(shape: Shape, blockSize: Int): Result<Unit> {
    val dims = shape.dims()
    if (dims.isEmpty()) {
        bail("scalar tensor cannot be quantized {shape:?}")
    }
    if (!dims[dims.len() - 1].isMultipleOf(blockSize)) {
        bail("quantized tensor must have their last dim divisible by block size {shape:?} {}", blockSize)
    }
    return Result.Ok(Unit)
}

public fun widenBf16(src: List<Bf16>): List<Float> {
    val out: List<Float> = ArrayList(src.size)
    val outPtr = out.asMutPtr().toInt()
    val nUnits = src.size.divCeil(WIDEN_CHUNK)
    barrierPool().executeChunked(nUnits, { range ->
        val outPtr1 = MutPtr(outPtr.toLong())
        for (unit in range) {
            val lo = unit * WIDEN_CHUNK
            val hi = src.size.min(lo + WIDEN_CHUNK)
            for ((i, v) in ((src).slice(lo until hi)).mapIndexed { index, item -> Pair(index, item) }) {
                (PrimitiveLayouts.F32).writeNative((MutPtr((outPtr1).rawAddress + ((lo + i).toLong() * (PrimitiveLayouts.F32).strideBytes))).rawAddress, v.toF32())
            }
        }
    })
    // SAFETY: every element written by exactly one unit.
    out.setLen(src.size)
    return out
}

public fun narrowBf16(src: List<Float>): List<Bf16> {
    val out: List<Bf16> = ArrayList(src.size)
    val outPtr = out.asMutPtr().toInt()
    val nUnits = src.size.divCeil(WIDEN_CHUNK)
    barrierPool().executeChunked(nUnits, { range ->
        val outPtr1 = MutPtr(outPtr.toLong())
        for (unit in range) {
            val lo = unit * WIDEN_CHUNK
            val hi = src.size.min(lo + WIDEN_CHUNK)
            for ((i, v) in ((src).slice(lo until hi)).mapIndexed { index, item -> Pair(index, item) }) {
                (ObjectSlotLayout<Bf16>()).writeNative((MutPtr((outPtr1).rawAddress + ((lo + i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress, fromF32(v))
            }
        }
    })
    // SAFETY: every element written by exactly one unit.
    out.setLen(src.size)
    return out
}

public fun Device.qzeros(elemCount: Int, dtype: GgmlDType): Result<QStorage> {
    return when (this) {
        Device.Cpu -> {
            val storage = dtype.cpuZeros(elemCount)
            Result.Ok(QStorage.Cpu(storage))
        }
        is Device.Metal -> {
            val (metal) = this
            val storage1 =
                when (val tryResult = QMetalStorage.zeros(metal, elemCount, dtype)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            Result.Ok(QStorage.Metal(storage1))
        }
        is Device.Cuda -> {
            val (cuda) = this
            val storage2 =
                when (val tryResult = QCudaStorage.zeros(cuda, elemCount, dtype)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }
            Result.Ok(QStorage.Cuda(storage2))
        }
    }
}

public fun <VecDotType, T : GgmlType> List<T>.matmulT(mkn: Triple<ULong, ULong, ULong>, lhs: List<Float>, dst: MutableList<Float>): Result<Unit> = matmul(mkn, lhs, this.asSlice(), dst)

public fun <VecDotType, T : GgmlType> List<T>.matmulTF16(mkn: Triple<ULong, ULong, ULong>, lhs: List<F16>, dst: MutableList<F16>): Result<Unit> = matmulF16(mkn, lhs, this.asSlice(), dst)

public fun <VecDotType, T : GgmlType> List<T>.embedding(ids: List<UInt>, rows: Int, hidden: Int, toFloatTFn: (Any?) -> Any?): Result<CpuStorage> {
    if (!hidden.isMultipleOf(T.BLCK_SIZE)) {
        bail("quantized embedding hidden size {hidden} is not divisible by block size {}", T.BLCK_SIZE)
    }
    val rowBlocks = hidden / T.BLCK_SIZE
    if (this.size != rows * rowBlocks) {
        bail("quantized tensor has {} blocks, expected {}", this.size, rows * rowBlocks)
    }
    val out = mutableListOf(0.0f, ids.size * hidden)
    for ((outRow, rowId) in ids.mapIndexed { index, item -> Pair(index, item) }) {
        val row = rowId.toInt()
        if (row >= rows) {
            bail("embedding id {row} is out of range for {rows} rows")
        }
        val src = ((this).slice(row * rowBlocks until (row + 1) * rowBlocks))
        val dst = ((out).slice(outRow * hidden until (outRow + 1) * hidden))
        toFloatTFn(src, dst)
    }
    return Result.Ok(CpuStorage.f32(out))
}

public fun <VecDotType, T : GgmlType> List<T>.size(typeLayoutT: TypeLayout<T>): Int = this.size * sizeOf<T>(typeLayoutT = typeLayoutT)

public fun <VecDotType, T : GgmlType> List<T>.fromFloat(xs: List<Float>, fromFloatTFn: (Any?) -> Any?) {
    fromFloatTFn(xs, this)
}

public fun <VecDotType, T : GgmlType> List<T>.fromFloatImatrix(xs: List<Float>, imatrixWeights: List<Float>, nPerRow: ULong, fromFloatImatrixTFn: (Any?) -> Any?) {
    fromFloatImatrixTFn(xs, this, imatrixWeights, nPerRow)
}

public fun <VecDotType, T : GgmlType> List<T>.dtype(): GgmlDType = T.DTYPE

public fun <VecDotType, T : GgmlType> List<T>.blockSize(): Int = T.BLCK_SIZE

public fun <VecDotType, T : GgmlType> List<T>.dequantize(elemCount: Int, toFloatTFn: (Any?) -> Any?): Result<CpuStorage> {
    val ys = mutableListOf(0.0f, elemCount)
    toFloatTFn(this.asSlice(), ys)
    return Result.Ok(CpuStorage.f32(ys))
}

public fun <VecDotType, T : GgmlType> List<T>.storageSizeInBytes(typeLayoutT: TypeLayout<T>): Int = this.size * sizeOf<T>(typeLayoutT = typeLayoutT)

public fun <VecDotType, T : GgmlType> List<T>.asPtrQuantizedType(): ConstPtr = this.asPtr()
