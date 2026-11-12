package io.github.kotlinmania.candle.cpu

import io.github.kotlinmania.candle.cpu.avx.vecAddBf16
import io.github.kotlinmania.candle.cpu.avx.vecAddF16
import io.github.kotlinmania.candle.cpu.avx.vecDotBf16
import io.github.kotlinmania.candle.cpu.avx.vecDotF16
import io.github.kotlinmania.candle.cpu.avx.vecDotF32
import io.github.kotlinmania.candle.cpu.avx.vecScalarAddBf16
import io.github.kotlinmania.candle.cpu.avx.vecScalarAddF16
import io.github.kotlinmania.candle.cpu.avx.vecSum
import io.github.kotlinmania.candle.cpu.features.get
import io.github.kotlinmania.candle.cpu.neon.vecMulBf16
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.rustlang.core.mem.ObjectSlotLayout
import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr

public fun scalarVecDotF32(aRow: ConstPtr, bRow: ConstPtr, c: MutPtr, k: Int) {
    val c1 = c
    for (i in 0 until k) {
        (c1).let { targetPtr ->
            val targetLayout = PrimitiveLayouts.F32
            var currentValue = targetLayout.readNative(targetPtr.rawAddress)
            currentValue += ((PrimitiveLayouts.F32).readNative((ConstPtr((aRow).rawAddress + ((i).toLong() * (PrimitiveLayouts.F32).strideBytes))).rawAddress) * (PrimitiveLayouts.F32).readNative((ConstPtr((bRow).rawAddress + ((i).toLong() * (PrimitiveLayouts.F32).strideBytes))).rawAddress))
            targetLayout.writeNative(targetPtr.rawAddress, currentValue)
        }
    }
}

public fun scalarVecSum(row: ConstPtr, b: MutPtr, k: Int) {
    val b1 = b
    (PrimitiveLayouts.F32).writeNative((b1).rawAddress, 0.0f)
    for (i in 0 until k) {
        (b1).let { targetPtr ->
            val targetLayout = PrimitiveLayouts.F32
            var currentValue = targetLayout.readNative(targetPtr.rawAddress)
            currentValue += ((PrimitiveLayouts.F32).readNative((ConstPtr((row).rawAddress + ((i).toLong() * (PrimitiveLayouts.F32).strideBytes))).rawAddress))
            targetLayout.writeNative(targetPtr.rawAddress, currentValue)
        }
    }
}

public fun scalarVecDotF16(aRow: ConstPtr, bRow: ConstPtr, c: MutPtr, k: Int) {
    val c1 = c
    var sum = 0.0
    for (i in 0 until k) {
        sum += (ObjectSlotLayout<F16>()).readNative((ConstPtr((aRow).rawAddress + ((i).toLong() * (ObjectSlotLayout<F16>()).strideBytes))).rawAddress).toF32() * (ObjectSlotLayout<F16>()).readNative((ConstPtr((bRow).rawAddress + ((i).toLong() * (ObjectSlotLayout<F16>()).strideBytes))).rawAddress).toF32()
    }
    (PrimitiveLayouts.F32).writeNative((c1).rawAddress, sum)
}

public fun scalarVecDotBf16(aRow: ConstPtr, bRow: ConstPtr, c: MutPtr, k: Int) {
    val c1 = c
    var sum = 0.0
    for (i in 0 until k) {
        sum += (ObjectSlotLayout<Bf16>()).readNative((ConstPtr((aRow).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress).toF32() * (ObjectSlotLayout<Bf16>()).readNative((ConstPtr((bRow).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress).toF32()
    }
    (PrimitiveLayouts.F32).writeNative((c1).rawAddress, sum)
}

public fun scalarVecAddF16(aRow: ConstPtr, bRow: ConstPtr, c: MutPtr, k: Int) {
    for (i in 0 until k) {
        (ObjectSlotLayout<F16>()).writeNative((MutPtr((c).rawAddress + ((i).toLong() * (ObjectSlotLayout<F16>()).strideBytes))).rawAddress, (ObjectSlotLayout<F16>()).readNative((ConstPtr((aRow).rawAddress + ((i).toLong() * (ObjectSlotLayout<F16>()).strideBytes))).rawAddress) + (ObjectSlotLayout<F16>()).readNative((ConstPtr((bRow).rawAddress + ((i).toLong() * (ObjectSlotLayout<F16>()).strideBytes))).rawAddress))
    }
}

public fun scalarVecAddBf16(aRow: ConstPtr, bRow: ConstPtr, c: MutPtr, k: Int) {
    for (i in 0 until k) {
        (ObjectSlotLayout<Bf16>()).writeNative((MutPtr((c).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress, (ObjectSlotLayout<Bf16>()).readNative((ConstPtr((aRow).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress) + (ObjectSlotLayout<Bf16>()).readNative((ConstPtr((bRow).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress))
    }
}

public fun scalarVecScalarAddF16(scalar: F16, xs: ConstPtr, ys: MutPtr, k: Int) {
    for (i in 0 until k) {
        (ObjectSlotLayout<F16>()).writeNative((MutPtr((ys).rawAddress + ((i).toLong() * (ObjectSlotLayout<F16>()).strideBytes))).rawAddress, (ObjectSlotLayout<F16>()).readNative((ConstPtr((xs).rawAddress + ((i).toLong() * (ObjectSlotLayout<F16>()).strideBytes))).rawAddress) + scalar)
    }
}

public fun scalarVecScalarAddBf16(scalar: Bf16, xs: ConstPtr, ys: MutPtr, k: Int) {
    for (i in 0 until k) {
        (ObjectSlotLayout<Bf16>()).writeNative((MutPtr((ys).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress, (ObjectSlotLayout<Bf16>()).readNative((ConstPtr((xs).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress) + scalar)
    }
}

public fun useAvx2(): Boolean = get().avx2

public fun useNeon(): Boolean = get().neon

public fun useAvx2F16(): Boolean {
    val features = get()
    return features.avx2 && (!false || features.f16c)
}

public fun useNeonF16(): Boolean {
    val features = get()
    return features.neon && (!false || features.fp16)
}

public fun useNeonBf16(): Boolean {
    val features = get()
    return features.neon && (!false || features.bf16)
}

internal fun vecDotF32(aRow: ConstPtr, bRow: ConstPtr, c: MutPtr, k: Int) {
    if (useAvx2()) {
        return vecDotF32(aRow, bRow, c, k)
    }
    if (useNeon()) {
        return io.github.kotlinmania.candle.cpu.neon
            .vecDotF32(aRow, bRow, c, k)
    }
    vecDotF32(aRow, bRow, c, k)
}

internal fun vecSum(row: ConstPtr, b: MutPtr, k: Int) {
    if (useAvx2()) {
        return vecSum(row, b, k)
    }
    if (useNeon()) {
        return io.github.kotlinmania.candle.cpu.neon
            .vecSum(row, b, k)
    }
    vecSum(row, b, k)
}

internal fun vecDotF16(aRow: ConstPtr, bRow: ConstPtr, c: MutPtr, k: Int) {
    if (useAvx2F16()) {
        return vecDotF16(aRow, bRow, c, k)
    }
    if (useNeonF16()) {
        return io.github.kotlinmania.candle.cpu.neon
            .vecDotF16(aRow, bRow, c, k)
    }
    scalarVecDotF16(aRow, bRow, c, k)
}

internal fun vecDotBf16(aRow: ConstPtr, bRow: ConstPtr, c: MutPtr, k: Int) {
    if (useAvx2()) {
        return vecDotBf16(aRow, bRow, c, k)
    }
    if (useNeonBf16()) {
        return io.github.kotlinmania.candle.cpu.neon
            .vecDotBf16(aRow, bRow, c, k)
    }
    scalarVecDotBf16(aRow, bRow, c, k)
}

internal fun vecAddF16(aRow: ConstPtr, bRow: ConstPtr, c: MutPtr, k: Int) {
    if (useAvx2F16()) {
        return vecAddF16(aRow, bRow, c, k)
    }
    if (useNeonF16()) {
        return io.github.kotlinmania.candle.cpu.neon
            .vecAddF16(aRow, bRow, c, k)
    }
    vecAddF16(aRow, bRow, c, k)
}

internal fun vecAddBf16(aRow: ConstPtr, bRow: ConstPtr, c: MutPtr, k: Int) {
    if (useAvx2()) {
        return vecAddBf16(aRow, bRow, c, k)
    }
    if (useNeonBf16()) {
        return io.github.kotlinmania.candle.cpu.neon
            .vecAddBf16(aRow, bRow, c, k)
    }
    vecAddBf16(aRow, bRow, c, k)
}

internal fun vecScalarAddF16(scalar: F16, xs: ConstPtr, ys: MutPtr, k: Int) {
    if (useAvx2F16()) {
        return vecScalarAddF16(scalar, xs, ys, k)
    }
    if (useNeonF16()) {
        return io.github.kotlinmania.candle.cpu.neon
            .vecScalarAddF16(scalar, xs, ys, k)
    }
    vecScalarAddF16(scalar, xs, ys, k)
}

internal fun vecMulBf16(aRow: ConstPtr, bRow: ConstPtr, c: MutPtr, k: Int) {
    if (useNeonBf16()) {
        return vecMulBf16(aRow, bRow, c, k)
    }
    for (j in 0 until k) {
        (ObjectSlotLayout<Bf16>()).writeNative((MutPtr((c).rawAddress + ((j).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress, (ObjectSlotLayout<Bf16>()).readNative((ConstPtr((aRow).rawAddress + ((j).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress) * (ObjectSlotLayout<Bf16>()).readNative((ConstPtr((bRow).rawAddress + ((j).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress))
    }
}

internal fun vecScalarAddBf16(scalar: Bf16, xs: ConstPtr, ys: MutPtr, k: Int) {
    if (useAvx2()) {
        return vecScalarAddBf16(scalar, xs, ys, k)
    }
    if (useNeonBf16()) {
        return io.github.kotlinmania.candle.cpu.neon
            .vecScalarAddBf16(scalar, xs, ys, k)
    }
    vecScalarAddBf16(scalar, xs, ys, k)
}
