package io.github.kotlinmania.candle

/**
 * Traits and methods for CPU-backed Tensors.
 *
 * Provides SIMD-accelerated vector addition, fused multiply-add, dot-product reduction,
 * activation functions, and type conversions for CPU tensor storage across f32, f16,
 * and bf16 elements on ARM NEON, x86 AVX2, and WebAssembly SIMD128 targets.
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
    // How many outer loop steps to accumulate in UnitType before flushing to f32.
    // Set to a finite value to bound rounding error (16 for example).
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

private val currentCpu = CpuNeon.CurrentCpu()
private val currentCpuF16 = CpuNeon.Fp16.Fallback.CurrentCpuF16()
private val currentCpuBF16 = CpuNeon.Bf16.Fallback.CurrentCpuBF16()
private val toF32: (Short) -> Float = { Float16Utils.f16BitsToFloat(it) }
private val bf16ToF32: (Short) -> Float = { Float16Utils.bf16BitsToFloat(it) }
private val addF16: (Short, Short) -> Short = { a, b ->
    Float16Utils.floatToF16Bits(Float16Utils.f16BitsToFloat(a) + Float16Utils.f16BitsToFloat(b))
}
private val addBf16: (Short, Short) -> Short = { a, b ->
    Float16Utils.floatToBF16Bits(Float16Utils.bf16BitsToFloat(a) + Float16Utils.bf16BitsToFloat(b))
}

public fun vecDotF32(aRow: FloatArray, bRow: FloatArray, c: FloatArray, k: Int) {
    val sum = currentCpu.zeroArray()

    var i = 0
    while (i + currentCpu.step <= k) {
        for (j in 0 until currentCpu.arr) {
            sum[j] = currentCpu.vecFma(
                sum[j],
                currentCpu.load(aRow, i + j * currentCpu.epr),
                currentCpu.load(bRow, i + j * currentCpu.epr)
            )
        }
        i += currentCpu.step
    }

    currentCpu.vecReduce(sum, c)

    // leftovers
    while (i < k) {
        c[0] += aRow[i] * bRow[i]
        i += 1
    }
}

public fun vecDotF32(aRow: FloatArray, bRow: FloatArray, c: FloatArray, k: UInt) {
    for (i in 0 until k.toInt()) {
        c[0] += aRow[i] * bRow[i]
    }
}

public fun vecSum(row: FloatArray, b: FloatArray, k: Int) {
    val np = k and (currentCpu.step - 1).inv()

    val sum = currentCpu.zeroArray()
    val x = currentCpu.zeroArray()

    for (i in 0 until np step currentCpu.step) {
        for (j in 0 until currentCpu.arr) {
            x[j] = currentCpu.load(row, i + j * currentCpu.epr)
            sum[j] = currentCpu.vecAdd(sum[j], x[j])
        }
    }

    currentCpu.vecReduce(sum, b)

    // leftovers
    for (i in np until k) {
        b[0] += row[i]
    }
}

public fun vecSum(row: FloatArray, b: FloatArray, k: UInt) {
    b[0] = 0.0f
    for (i in 0 until k.toInt()) {
        b[0] += row[i]
    }
}

public fun vecDotF16(aRow: ShortArray, bRow: ShortArray, c: FloatArray, k: Int) {
    var sumf = 0.0f
    var sum = currentCpuF16.zeroArray()
    var stepsSinceFlush = 0
    var i = 0
    while (i + currentCpuF16.step <= k) {
        for (j in 0 until currentCpuF16.arr) {
            sum[j] = currentCpuF16.vecFma(
                sum[j],
                currentCpuF16.load(aRow, i + j * currentCpuF16.epr),
                currentCpuF16.load(bRow, i + j * currentCpuF16.epr)
            )
        }
        stepsSinceFlush += 1
        if (stepsSinceFlush == currentCpuF16.flushInterval) {
            val partial = floatArrayOf(0.0f)
            currentCpuF16.vecReduce(sum, partial)
            sumf += partial[0]
            sum = currentCpuF16.zeroArray()
            stepsSinceFlush = 0
        }

        i += currentCpuF16.step
    }

    val partial = floatArrayOf(0.0f)
    currentCpuF16.vecReduce(sum, partial)
    sumf += partial[0]

    // leftovers
    while (i < k) {
        sumf += toF32(aRow[i]) * toF32(bRow[i])
        i += 1
    }
    c[0] = sumf
}

public fun vecDotBf16(aRow: ShortArray, bRow: ShortArray, c: FloatArray, k: Int) {
    val sum = currentCpuBF16.zeroArray()

    var i = 0
    while (i + currentCpuBF16.step <= k) {
        for (j in 0 until currentCpuBF16.arr) {
            sum[j] = currentCpuBF16.vecFma(
                sum[j],
                currentCpuBF16.load(aRow, i + j * currentCpuBF16.epr),
                currentCpuBF16.load(bRow, i + j * currentCpuBF16.epr)
            )
        }
        i += currentCpuBF16.step
    }

    currentCpuBF16.vecReduce(sum, c)

    // leftovers
    while (i < k) {
        c[0] += bf16ToF32(aRow[i]) * bf16ToF32(bRow[i])
        i += 1
    }
}

public fun vecDotF16(aRow: ShortArray, bRow: ShortArray, c: FloatArray, k: UInt) {
    // leftovers
    var sum = 0.0f
    for (i in 0 until k.toInt()) {
        sum += toF32(aRow[i]) * toF32(bRow[i])
    }
    c[0] = sum
}

public fun vecDotBf16(aRow: ShortArray, bRow: ShortArray, c: FloatArray, k: UInt) {
    // leftovers
    var sum = 0.0f
    for (i in 0 until k.toInt()) {
        sum += bf16ToF32(aRow[i]) * bf16ToF32(bRow[i])
    }
    c[0] = sum
}

public fun vecAddF16(aRow: ShortArray, bRow: ShortArray, c: ShortArray, k: Int) {
    var i = 0
    while (i + currentCpuF16.step <= k) {
        for (j in 0 until currentCpuF16.arr) {
            currentCpuF16.vecStore(
                c, i + j * currentCpuF16.epr,
                currentCpuF16.vecAdd(
                    currentCpuF16.load(aRow, i + j * currentCpuF16.epr),
                    currentCpuF16.load(bRow, i + j * currentCpuF16.epr)
                )
            )
        }
        i += currentCpuF16.step
    }

    // leftovers
    for (j in i until k) {
        c[j] = addF16(aRow[j], bRow[j])
    }
}

public fun vecAddF16(aRow: ShortArray, bRow: ShortArray, c: ShortArray, k: UInt) {
    for (i in 0 until k.toInt()) {
        c[i] = addF16(aRow[i], bRow[i])
    }
}

public fun vecAddBf16(aRow: ShortArray, bRow: ShortArray, c: ShortArray, k: Int) {
    var i = 0
    while (i + currentCpuBF16.step <= k) {
        for (j in 0 until currentCpuBF16.arr) {
            currentCpuBF16.vecStore(
                c, i + j * currentCpuBF16.epr,
                currentCpuBF16.vecAdd(
                    currentCpuBF16.load(aRow, i + j * currentCpuBF16.epr),
                    currentCpuBF16.load(bRow, i + j * currentCpuBF16.epr)
                )
            )
        }
        i += currentCpuBF16.step
    }

    // leftovers
    for (j in i until k) {
        c[j] = addBf16(aRow[j], bRow[j])
    }
}

public fun vecAddBf16(aRow: ShortArray, bRow: ShortArray, c: ShortArray, k: UInt) {
    for (i in 0 until k.toInt()) {
        c[i] = addBf16(aRow[i], bRow[i])
    }
}

public fun vecScalarAddF16(scalar: Short, xs: ShortArray, ys: ShortArray, k: Int) {
    val sv = currentCpuF16.fromF32(toF32(scalar))
    var i = 0
    while (i + currentCpuF16.step <= k) {
        for (j in 0 until currentCpuF16.arr) {
            currentCpuF16.vecStore(
                ys, i + j * currentCpuF16.epr,
                currentCpuF16.vecAdd(currentCpuF16.load(xs, i + j * currentCpuF16.epr), sv)
            )
        }
        i += currentCpuF16.step
    }
    for (j in i until k) {
        ys[j] = addF16(xs[j], scalar)
    }
}

public fun vecScalarAddF16(scalar: Short, xs: ShortArray, ys: ShortArray, k: UInt) {
    for (i in 0 until k.toInt()) {
        ys[i] = addF16(xs[i], scalar)
    }
}

public fun vecScalarAddBf16(scalar: Short, xs: ShortArray, ys: ShortArray, k: Int) {
    val sv = currentCpuBF16.fromF32(bf16ToF32(scalar))
    var i = 0
    while (i + currentCpuBF16.step <= k) {
        for (j in 0 until currentCpuBF16.arr) {
            currentCpuBF16.vecStore(
                ys, i + j * currentCpuBF16.epr,
                currentCpuBF16.vecAdd(
                    currentCpuBF16.load(xs, i + j * currentCpuBF16.epr),
                    sv
                )
            )
        }
        i += currentCpuBF16.step
    }
    for (j in i until k) {
        ys[j] = addBf16(xs[j], scalar)
    }
}

public fun vecScalarAddBf16(scalar: Short, xs: ShortArray, ys: ShortArray, k: UInt) {
    for (i in 0 until k.toInt()) {
        ys[i] = addBf16(xs[i], scalar)
    }
}
