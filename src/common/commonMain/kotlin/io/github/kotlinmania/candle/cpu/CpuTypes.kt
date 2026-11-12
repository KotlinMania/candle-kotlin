package io.github.kotlinmania.candle.cpu

import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr

public interface Cpu<Unit, Array> {
    public val STEP: Int

    public val EPR: Int

    public val ARR: Int
        get() = Cpu.STEP / Cpu.EPR

    public fun zero()

    public fun zeroArray(): Array

    public fun load(memAddr: ConstPtr<Float>)

    public fun vecAdd(a: Unit, b: Unit)

    public fun vecFma(a: Unit, b: Unit, c: Unit)

    public fun vecReduce(x: Array, y: MutPtr<Float>)

    public fun fromF32(v: Float)

    public fun vecStore(memAddr: MutPtr<Float>, a: Unit)
}

public interface CpuF16<Unit, Array> {
    public val STEP: Int

    public val EPR: Int

    public val ARR: Int
        get() = CpuF16.STEP / CpuF16.EPR

    public val FLUSH_INTERVAL: Int
        get() = Int.MAX_VALUE

    public fun zero()

    public fun zeroArray(): Array

    public fun load(memAddr: ConstPtr<F16>)

    public fun vecAdd(a: Unit, b: Unit)

    public fun vecFma(a: Unit, b: Unit, c: Unit)

    public fun vecReduce(x: Array, y: MutPtr<Float>)

    public fun fromF32(v: Float)

    public fun vecStore(memAddr: MutPtr<F16>, a: Unit)
}

public interface CpuBF16<Unit, Array> {
    public val STEP: Int

    public val EPR: Int

    public val ARR: Int
        get() = CpuBF16.STEP / CpuBF16.EPR

    public fun zero()

    public fun zeroArray(): Array

    public fun load(memAddr: ConstPtr<Bf16>)

    public fun vecAdd(a: Unit, b: Unit)

    public fun vecFma(a: Unit, b: Unit, c: Unit)

    public fun vecReduce(x: Array, y: MutPtr<Float>)

    public fun fromF32(v: Float)

    public fun vecStore(memAddr: MutPtr<Bf16>, a: Unit)
}
