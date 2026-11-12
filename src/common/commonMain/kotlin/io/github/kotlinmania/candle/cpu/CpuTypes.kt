package io.github.kotlinmania.candle.cpu

import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16

public interface Cpu<Unit, Array> {
    public val STEP: Int

    public val EPR: Int

    public val ARR: Int
        get() = Cpu.STEP / Cpu.EPR

    public fun zero()

    public fun zeroArray(): Array

    public fun load(memAddr: org.rustlang.runtime.RustPointer<Float>?)

    public fun vecAdd(a: Unit, b: Unit)

    public fun vecFma(a: Unit, b: Unit, c: Unit)

    public fun vecReduce(x: Array, y: org.rustlang.runtime.RustPointer<Float>?)

    public fun fromF32(v: Float)

    public fun vecStore(memAddr: org.rustlang.runtime.RustPointer<Float>?, a: Unit)
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

    public fun load(memAddr: org.rustlang.runtime.RustPointer<F16>?)

    public fun vecAdd(a: Unit, b: Unit)

    public fun vecFma(a: Unit, b: Unit, c: Unit)

    public fun vecReduce(x: Array, y: org.rustlang.runtime.RustPointer<Float>?)

    public fun fromF32(v: Float)

    public fun vecStore(memAddr: org.rustlang.runtime.RustPointer<F16>?, a: Unit)
}

public interface CpuBF16<Unit, Array> {
    public val STEP: Int

    public val EPR: Int

    public val ARR: Int
        get() = CpuBF16.STEP / CpuBF16.EPR

    public fun zero()

    public fun zeroArray(): Array

    public fun load(memAddr: org.rustlang.runtime.RustPointer<Bf16>?)

    public fun vecAdd(a: Unit, b: Unit)

    public fun vecFma(a: Unit, b: Unit, c: Unit)

    public fun vecReduce(x: Array, y: org.rustlang.runtime.RustPointer<Float>?)

    public fun fromF32(v: Float)

    public fun vecStore(memAddr: org.rustlang.runtime.RustPointer<Bf16>?, a: Unit)
}
