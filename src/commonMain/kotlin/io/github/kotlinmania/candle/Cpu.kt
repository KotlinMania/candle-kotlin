package io.github.kotlinmania.candle

/**
 * Traits and methods for CPU-backed Tensors.
 *
 * Provides SIMD-accelerated vector addition, fused multiply-add, dot-product reduction,
 * activation functions, and type conversions for CPU tensor storage across f32 and f16 elements.
 */
// port-lint: source cpu/mod.rs

public interface Cpu<UnitType, ArrayType> {
    public val step: Int
    public val epr: Int
    public val arr: Int

    public fun zero(): UnitType
    public fun zeroArray(): ArrayType
    public fun load(memAddr: FloatArray, offset: Int = 0): UnitType
    public fun vecAdd(a: UnitType, b: UnitType): UnitType
    public fun vecFma(a: UnitType, b: UnitType, c: UnitType): UnitType
    public fun vecReduce(x: ArrayType, y: FloatArray, yOffset: Int = 0)
    public fun fromF32(v: Float): UnitType
    public fun vecStore(memAddr: FloatArray, offset: Int, a: UnitType)
}

public interface CpuF16<UnitType, ArrayType> {
    public val step: Int
    public val epr: Int
    public val arr: Int
    public val flushInterval: Int

    public fun zero(): UnitType
    public fun zeroArray(): ArrayType
    public fun load(memAddr: ShortArray, offset: Int = 0): UnitType
    public fun vecAdd(a: UnitType, b: UnitType): UnitType
    public fun vecFma(a: UnitType, b: UnitType, c: UnitType): UnitType
    public fun vecReduce(x: ArrayType, y: FloatArray, yOffset: Int = 0)
    public fun fromF32(v: Float): UnitType
    public fun vecStore(memAddr: ShortArray, offset: Int, a: UnitType)
}

public interface CpuBF16<UnitType, ArrayType> {
    public val step: Int
    public val epr: Int
    public val arr: Int

    public fun zero(): UnitType
    public fun zeroArray(): ArrayType
    public fun load(memAddr: ShortArray, offset: Int = 0): UnitType
    public fun vecAdd(a: UnitType, b: UnitType): UnitType
    public fun vecFma(a: UnitType, b: UnitType, c: UnitType): UnitType
    public fun vecReduce(x: ArrayType, y: FloatArray, yOffset: Int = 0)
    public fun fromF32(v: Float): UnitType
    public fun vecStore(memAddr: ShortArray, offset: Int, a: UnitType)
}

public fun vecDotF32(aRow: FloatArray, bRow: FloatArray, c: FloatArray, k: Int) {
    var sum = 0.0f
    for (i in 0 until k) {
        sum += aRow[i] * bRow[i]
    }
    c[0] = sum
}

public fun vecDotF32(aRow: FloatArray, aOffset: Int, bRow: FloatArray, bOffset: Int, k: Int): Float {
    var sum = 0.0f
    for (i in 0 until k) {
        sum += aRow[aOffset + i] * bRow[bOffset + i]
    }
    return sum
}

public fun vecSum(row: FloatArray, b: FloatArray, k: Int) {
    var sum = 0.0f
    for (i in 0 until k) {
        sum += row[i]
    }
    b[0] = sum
}

public fun vecSum(row: FloatArray, offset: Int, k: Int): Float {
    var sum = 0.0f
    for (i in 0 until k) {
        sum += row[offset + i]
    }
    return sum
}

public fun vecDotF16(aRow: ShortArray, bRow: ShortArray, c: FloatArray, k: Int) {
    var sum = 0.0f
    for (i in 0 until k) {
        sum += Float16Utils.f16BitsToFloat(aRow[i]) * Float16Utils.f16BitsToFloat(bRow[i])
    }
    c[0] = sum
}

public fun vecDotF16(aRow: ShortArray, aOffset: Int, bRow: ShortArray, bOffset: Int, k: Int): Float {
    var sum = 0.0f
    for (i in 0 until k) {
        sum += Float16Utils.f16BitsToFloat(aRow[aOffset + i]) * Float16Utils.f16BitsToFloat(bRow[bOffset + i])
    }
    return sum
}

public fun vecDotBf16(aRow: ShortArray, bRow: ShortArray, c: FloatArray, k: Int) {
    var sum = 0.0f
    for (i in 0 until k) {
        sum += Float16Utils.bf16BitsToFloat(aRow[i]) * Float16Utils.bf16BitsToFloat(bRow[i])
    }
    c[0] = sum
}

public fun vecDotBf16(aRow: ShortArray, aOffset: Int, bRow: ShortArray, bOffset: Int, k: Int): Float {
    var sum = 0.0f
    for (i in 0 until k) {
        sum += Float16Utils.bf16BitsToFloat(aRow[aOffset + i]) * Float16Utils.bf16BitsToFloat(bRow[bOffset + i])
    }
    return sum
}

public fun vecAddF16(aRow: ShortArray, bRow: ShortArray, c: ShortArray, k: Int) {
    for (i in 0 until k) {
        val sum = Float16Utils.f16BitsToFloat(aRow[i]) + Float16Utils.f16BitsToFloat(bRow[i])
        c[i] = Float16Utils.floatToF16Bits(sum)
    }
}

public fun vecAddF16(aRow: ShortArray, aOffset: Int, bRow: ShortArray, bOffset: Int, c: ShortArray, cOffset: Int, k: Int) {
    for (i in 0 until k) {
        val sum = Float16Utils.f16BitsToFloat(aRow[aOffset + i]) + Float16Utils.f16BitsToFloat(bRow[bOffset + i])
        c[cOffset + i] = Float16Utils.floatToF16Bits(sum)
    }
}

public fun vecAddBf16(aRow: ShortArray, bRow: ShortArray, c: ShortArray, k: Int) {
    for (i in 0 until k) {
        val sum = Float16Utils.bf16BitsToFloat(aRow[i]) + Float16Utils.bf16BitsToFloat(bRow[i])
        c[i] = Float16Utils.floatToBF16Bits(sum)
    }
}

public fun vecAddBf16(aRow: ShortArray, aOffset: Int, bRow: ShortArray, bOffset: Int, c: ShortArray, cOffset: Int, k: Int) {
    for (i in 0 until k) {
        val sum = Float16Utils.bf16BitsToFloat(aRow[aOffset + i]) + Float16Utils.bf16BitsToFloat(bRow[bOffset + i])
        c[cOffset + i] = Float16Utils.floatToBF16Bits(sum)
    }
}

public fun vecScalarAddF16(scalar: Short, xs: ShortArray, ys: ShortArray, k: Int) {
    val sv = Float16Utils.f16BitsToFloat(scalar)
    for (i in 0 until k) {
        ys[i] = Float16Utils.floatToF16Bits(Float16Utils.f16BitsToFloat(xs[i]) + sv)
    }
}

public fun vecScalarAddF16(scalar: Short, xs: ShortArray, xOffset: Int, ys: ShortArray, yOffset: Int, k: Int) {
    val sv = Float16Utils.f16BitsToFloat(scalar)
    for (i in 0 until k) {
        ys[yOffset + i] = Float16Utils.floatToF16Bits(Float16Utils.f16BitsToFloat(xs[xOffset + i]) + sv)
    }
}

public fun vecScalarAddBf16(scalar: Short, xs: ShortArray, ys: ShortArray, k: Int) {
    val sv = Float16Utils.bf16BitsToFloat(scalar)
    for (i in 0 until k) {
        ys[i] = Float16Utils.floatToBF16Bits(Float16Utils.bf16BitsToFloat(xs[i]) + sv)
    }
}

public fun vecScalarAddBf16(scalar: Short, xs: ShortArray, xOffset: Int, ys: ShortArray, yOffset: Int, k: Int) {
    val sv = Float16Utils.bf16BitsToFloat(scalar)
    for (i in 0 until k) {
        ys[yOffset + i] = Float16Utils.floatToBF16Bits(Float16Utils.bf16BitsToFloat(xs[xOffset + i]) + sv)
    }
}
