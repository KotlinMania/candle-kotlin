package io.github.kotlinmania.candle

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
