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

public fun scalarVecDotF32(aRow: org.rustlang.runtime.RustPointer<Float>?, bRow: org.rustlang.runtime.RustPointer<Float>?, c: org.rustlang.runtime.RustPointer<Float>?, k: Int) {
    val c1 = c
    for (i in 0 until k) {
        (c1)!!.let { __rustPlace -> var __rustValue = __rustPlace.read(); __rustValue += ((org.rustlang.runtime.rustPointerOffset(aRow, null, (i).toLong(), false, false))!!.read() * (org.rustlang.runtime.rustPointerOffset(bRow, null, (i).toLong(), false, false))!!.read()); __rustPlace.write(__rustValue) }
    }
}

public fun scalarVecSum(row: org.rustlang.runtime.RustPointer<Float>?, b: org.rustlang.runtime.RustPointer<Float>?, k: Int) {
    val b1 = b
    (b1)!!.write(0.0f)
    for (i in 0 until k) {
        (b1)!!.let { __rustPlace -> var __rustValue = __rustPlace.read(); __rustValue += ((org.rustlang.runtime.rustPointerOffset(row, null, (i).toLong(), false, false))!!.read()); __rustPlace.write(__rustValue) }
    }
}

public fun scalarVecDotF16(aRow: org.rustlang.runtime.RustPointer<F16>?, bRow: org.rustlang.runtime.RustPointer<F16>?, c: org.rustlang.runtime.RustPointer<Float>?, k: Int) {
    val c1 = c
    var sum = 0.0
    for (i in 0 until k) {
        sum += (org.rustlang.runtime.rustPointerOffset(aRow, null, (i).toLong(), false, false))!!.read().toF32() * (org.rustlang.runtime.rustPointerOffset(bRow, null, (i).toLong(), false, false))!!.read().toF32()
    }
    (c1)!!.write(sum)
}

public fun scalarVecDotBf16(aRow: org.rustlang.runtime.RustPointer<Bf16>?, bRow: org.rustlang.runtime.RustPointer<Bf16>?, c: org.rustlang.runtime.RustPointer<Float>?, k: Int) {
    val c1 = c
    var sum = 0.0
    for (i in 0 until k) {
        sum += (org.rustlang.runtime.rustPointerOffset(aRow, null, (i).toLong(), false, false))!!.read().toF32() * (org.rustlang.runtime.rustPointerOffset(bRow, null, (i).toLong(), false, false))!!.read().toF32()
    }
    (c1)!!.write(sum)
}

public fun scalarVecAddF16(aRow: org.rustlang.runtime.RustPointer<F16>?, bRow: org.rustlang.runtime.RustPointer<F16>?, c: org.rustlang.runtime.RustPointer<F16>?, k: Int) {
    for (i in 0 until k) {
        (org.rustlang.runtime.rustPointerOffset(c, null, (i).toLong(), false, false))!!.write((org.rustlang.runtime.rustPointerOffset(aRow, null, (i).toLong(), false, false))!!.read() + (org.rustlang.runtime.rustPointerOffset(bRow, null, (i).toLong(), false, false))!!.read())
    }
}

public fun scalarVecAddBf16(aRow: org.rustlang.runtime.RustPointer<Bf16>?, bRow: org.rustlang.runtime.RustPointer<Bf16>?, c: org.rustlang.runtime.RustPointer<Bf16>?, k: Int) {
    for (i in 0 until k) {
        (org.rustlang.runtime.rustPointerOffset(c, null, (i).toLong(), false, false))!!.write((org.rustlang.runtime.rustPointerOffset(aRow, null, (i).toLong(), false, false))!!.read() + (org.rustlang.runtime.rustPointerOffset(bRow, null, (i).toLong(), false, false))!!.read())
    }
}

public fun scalarVecScalarAddF16(scalar: F16, xs: org.rustlang.runtime.RustPointer<F16>?, ys: org.rustlang.runtime.RustPointer<F16>?, k: Int) {
    for (i in 0 until k) {
        (org.rustlang.runtime.rustPointerOffset(ys, null, (i).toLong(), false, false))!!.write((org.rustlang.runtime.rustPointerOffset(xs, null, (i).toLong(), false, false))!!.read() + scalar)
    }
}

public fun scalarVecScalarAddBf16(scalar: Bf16, xs: org.rustlang.runtime.RustPointer<Bf16>?, ys: org.rustlang.runtime.RustPointer<Bf16>?, k: Int) {
    for (i in 0 until k) {
        (org.rustlang.runtime.rustPointerOffset(ys, null, (i).toLong(), false, false))!!.write((org.rustlang.runtime.rustPointerOffset(xs, null, (i).toLong(), false, false))!!.read() + scalar)
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

internal fun vecDotF32(aRow: org.rustlang.runtime.RustPointer<Float>?, bRow: org.rustlang.runtime.RustPointer<Float>?, c: org.rustlang.runtime.RustPointer<Float>?, k: Int) {
    if (useAvx2()) {
        return vecDotF32(aRow, bRow, c, k)
    }
    if (useNeon()) {
        return io.github.kotlinmania.candle.cpu.neon.vecDotF32(aRow, bRow, c, k)
    }
    vecDotF32(aRow, bRow, c, k)
}

internal fun vecSum(row: org.rustlang.runtime.RustPointer<Float>?, b: org.rustlang.runtime.RustPointer<Float>?, k: Int) {
    if (useAvx2()) {
        return vecSum(row, b, k)
    }
    if (useNeon()) {
        return io.github.kotlinmania.candle.cpu.neon.vecSum(row, b, k)
    }
    vecSum(row, b, k)
}

internal fun vecDotF16(aRow: org.rustlang.runtime.RustPointer<F16>?, bRow: org.rustlang.runtime.RustPointer<F16>?, c: org.rustlang.runtime.RustPointer<Float>?, k: Int) {
    if (useAvx2F16()) {
        return vecDotF16(aRow, bRow, c, k)
    }
    if (useNeonF16()) {
        return io.github.kotlinmania.candle.cpu.neon.vecDotF16(aRow, bRow, c, k)
    }
    scalarVecDotF16(aRow, bRow, c, k)
}

internal fun vecDotBf16(aRow: org.rustlang.runtime.RustPointer<Bf16>?, bRow: org.rustlang.runtime.RustPointer<Bf16>?, c: org.rustlang.runtime.RustPointer<Float>?, k: Int) {
    if (useAvx2()) {
        return vecDotBf16(aRow, bRow, c, k)
    }
    if (useNeonBf16()) {
        return io.github.kotlinmania.candle.cpu.neon.vecDotBf16(aRow, bRow, c, k)
    }
    scalarVecDotBf16(aRow, bRow, c, k)
}

internal fun vecAddF16(aRow: org.rustlang.runtime.RustPointer<F16>?, bRow: org.rustlang.runtime.RustPointer<F16>?, c: org.rustlang.runtime.RustPointer<F16>?, k: Int) {
    if (useAvx2F16()) {
        return vecAddF16(aRow, bRow, c, k)
    }
    if (useNeonF16()) {
        return io.github.kotlinmania.candle.cpu.neon.vecAddF16(aRow, bRow, c, k)
    }
    vecAddF16(aRow, bRow, c, k)
}

internal fun vecAddBf16(aRow: org.rustlang.runtime.RustPointer<Bf16>?, bRow: org.rustlang.runtime.RustPointer<Bf16>?, c: org.rustlang.runtime.RustPointer<Bf16>?, k: Int) {
    if (useAvx2()) {
        return vecAddBf16(aRow, bRow, c, k)
    }
    if (useNeonBf16()) {
        return io.github.kotlinmania.candle.cpu.neon.vecAddBf16(aRow, bRow, c, k)
    }
    vecAddBf16(aRow, bRow, c, k)
}

internal fun vecScalarAddF16(scalar: F16, xs: org.rustlang.runtime.RustPointer<F16>?, ys: org.rustlang.runtime.RustPointer<F16>?, k: Int) {
    if (useAvx2F16()) {
        return vecScalarAddF16(scalar, xs, ys, k)
    }
    if (useNeonF16()) {
        return io.github.kotlinmania.candle.cpu.neon.vecScalarAddF16(scalar, xs, ys, k)
    }
    vecScalarAddF16(scalar, xs, ys, k)
}

internal fun vecMulBf16(aRow: org.rustlang.runtime.RustPointer<Bf16>?, bRow: org.rustlang.runtime.RustPointer<Bf16>?, c: org.rustlang.runtime.RustPointer<Bf16>?, k: Int) {
    if (useNeonBf16()) {
        return vecMulBf16(aRow, bRow, c, k)
    }
    for (j in 0 until k) {
        (org.rustlang.runtime.rustPointerOffset(c, null, (j).toLong(), false, false))!!.write((org.rustlang.runtime.rustPointerOffset(aRow, null, (j).toLong(), false, false))!!.read() * (org.rustlang.runtime.rustPointerOffset(bRow, null, (j).toLong(), false, false))!!.read())
    }
}

internal fun vecScalarAddBf16(scalar: Bf16, xs: org.rustlang.runtime.RustPointer<Bf16>?, ys: org.rustlang.runtime.RustPointer<Bf16>?, k: Int) {
    if (useAvx2()) {
        return vecScalarAddBf16(scalar, xs, ys, k)
    }
    if (useNeonBf16()) {
        return io.github.kotlinmania.candle.cpu.neon.vecScalarAddBf16(scalar, xs, ys, k)
    }
    vecScalarAddBf16(scalar, xs, ys, k)
}
