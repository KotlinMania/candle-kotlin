// port-lint: source cpu/simd128.rs
package io.github.kotlinmania.candle.cpu.simd128

import io.github.kotlinmania.candle.cpu.Cpu
import io.github.kotlinmania.candle.cpu.CpuBF16
import io.github.kotlinmania.candle.cpu.CpuF16
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.rustlang.core.arch.wasm32.*
import io.github.kotlinmania.rustlang.core.mem.ObjectSlotLayout
import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.mem.TypeLayout
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr

public class CurrentCpu : Cpu {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<CurrentCpu> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<CurrentCpu> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): CurrentCpu =
                    CurrentCpu()
                override fun writeNative(addr: Long, value: CurrentCpu) {
                }
            }

        public val STEP: Int = STEP

        public val EPR: Int = EPR

        public val ARR: Int = ARR

        public fun zero(): V128 = f32x4Splat(0.0)

        public fun zeroArray(): List<V128> = List(ARR) { CurrentCpu.zero() }

        public fun fromF32(v: Float): V128 = f32x4Splat(v)

        public fun load(memAddr: ConstPtr): V128 = v128Load(MutPtr(memAddr.rawAddress, memAddr.rawMeta))

        public fun vecAdd(a: V128, b: V128): V128 = f32x4Add(a, b)

        public fun vecFma(a: V128, b: V128, c: V128): V128 = f32x4Add(f32x4Mul(b, c), a)

        public fun vecStore(memAddr: MutPtr, a: V128) {
            v128Store(MutPtr(memAddr.rawAddress, memAddr.rawMeta), a)
        }

        public fun vecReduce(x: List<V128>, y: MutPtr) {
            val y1 = y
            for (i in 0 until ARR / 2) {
                x[2 * i] = f32x4Add(x[2 * i], x[2 * i + 1])
            }
            for (i1 in 0 until ARR / 4) {
                x[4 * i1] = f32x4Add(x[4 * i1], x[4 * i1 + 2])
            }
            for (i2 in 0 until ARR / 8) {
                x[8 * i2] = f32x4Add(x[8 * i2], x[8 * i2 + 4])
            }
            (PrimitiveLayouts.F32).writeNative((y1).rawAddress, f32x4ExtractLane(x[0], 0) + f32x4ExtractLane(x[0], 1) + f32x4ExtractLane(x[0], 2) + f32x4ExtractLane(x[0], 3))
        }
    }
}

public val STEP: Int = 16

public val EPR: Int = 4

public val ARR: Int = STEP / EPR

// WebAssembly SIMD has no half-precision arithmetic, so widen f16 lanes to
// f32x4 as the AVX software fallback and the NEON fallback do.
public class CurrentCpuF16 : CpuF16 {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<CurrentCpuF16> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<CurrentCpuF16> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): CurrentCpuF16 =
                    CurrentCpuF16()
                override fun writeNative(addr: Long, value: CurrentCpuF16) {
                }
            }

        public val STEP: Int = STEP

        public val EPR: Int = EPR

        public val ARR: Int = ARR

        public fun zero(): V128 = CurrentCpu.zero()

        public fun zeroArray(): List<V128> = CurrentCpu.zeroArray()

        public fun fromF32(v: Float): V128 = CurrentCpu.fromF32(v)

        public fun load(memAddr: ConstPtr): V128 {
            val values = listOf((ObjectSlotLayout<F16>()).readNative((memAddr).rawAddress).toF32(), (ObjectSlotLayout<F16>()).readNative((ConstPtr((memAddr).rawAddress + ((1).toLong() * (ObjectSlotLayout<F16>()).strideBytes))).rawAddress).toF32(), (ObjectSlotLayout<F16>()).readNative((ConstPtr((memAddr).rawAddress + ((2).toLong() * (ObjectSlotLayout<F16>()).strideBytes))).rawAddress).toF32(), (ObjectSlotLayout<F16>()).readNative((ConstPtr((memAddr).rawAddress + ((3).toLong() * (ObjectSlotLayout<F16>()).strideBytes))).rawAddress).toF32())
            return v128Load(values.asPtr().cast())
        }

        public fun vecAdd(a: V128, b: V128): V128 = CurrentCpu.vecAdd(a, b)

        public fun vecFma(a: V128, b: V128, c: V128): V128 = CurrentCpu.vecFma(a, b, c)

        public fun vecStore(memAddr: MutPtr, a: V128) {
            val values = List(EPR) { 0.0f }
            v128Store(values.asMutPtr().cast(), a)
            for ((i, value) in values.intoIter().mapIndexed { index, item -> Pair(index, item) }) {
                (ObjectSlotLayout<F16>()).writeNative((MutPtr((memAddr).rawAddress + ((i).toLong() * (ObjectSlotLayout<F16>()).strideBytes))).rawAddress, F16.fromF32(value))
            }
        }

        public fun vecReduce(x: List<V128>, y: MutPtr) {
            CurrentCpu.vecReduce(x, y)
        }
    }
}

// bf16 is the upper half of an f32. Widen its bits before doing arithmetic in
// f32x4, then use half's scalar conversion to preserve round-to-even on store.
public class CurrentCpuBF16 : CpuBF16 {
    public companion object {
        public fun typeLayout(): io.github.kotlinmania.rustlang.core.mem.TypeLayout<CurrentCpuBF16> =
            object : io.github.kotlinmania.rustlang.core.mem.TypeLayout<CurrentCpuBF16> {
                override val alignBytes: Long = 1L
                override val sizeBytes: Long = 0L
                override fun readNative(addr: Long): CurrentCpuBF16 =
                    CurrentCpuBF16()
                override fun writeNative(addr: Long, value: CurrentCpuBF16) {
                }
            }

        public val STEP: Int = STEP

        public val EPR: Int = EPR

        public val ARR: Int = ARR

        public fun zero(): V128 = CurrentCpu.zero()

        public fun zeroArray(): List<V128> = CurrentCpu.zeroArray()

        public fun fromF32(v: Float): V128 = CurrentCpu.fromF32(v)

        public fun load(memAddr: ConstPtr): V128 {
            val values = v128Load64Zero(memAddr)
            return u32x4Shl(u32x4ExtendLowU16x8(values), 16)
        }

        public fun vecAdd(a: V128, b: V128): V128 = CurrentCpu.vecAdd(a, b)

        public fun vecFma(a: V128, b: V128, c: V128): V128 = CurrentCpu.vecFma(a, b, c)

        public fun vecStore(memAddr: MutPtr, a: V128) {
            val values = List(EPR) { 0.0f }
            v128Store(values.asMutPtr().cast(), a)
            for ((i, value) in values.intoIter().mapIndexed { index, item -> Pair(index, item) }) {
                (ObjectSlotLayout<Bf16>()).writeNative((MutPtr((memAddr).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress, Bf16.fromF32(value))
            }
        }

        public fun vecReduce(x: List<V128>, y: MutPtr) {
            CurrentCpu.vecReduce(x, y)
        }
    }
}

internal fun vecDotF32(aRow: ConstPtr, bRow: ConstPtr, c: MutPtr, k: Int) {
    val c1 = c
    val sum = CurrentCpu.zeroArray()
    var i = 0
    while (i + CurrentCpu.STEP <= k) {
        for (j in 0 until CurrentCpu.ARR) {
            sum[j] = CurrentCpu.vecFma(sum[j], CurrentCpu.load(ConstPtr((aRow).rawAddress + ((i + j * CurrentCpu.EPR).toLong() * (PrimitiveLayouts.F32).strideBytes))), CurrentCpu.load(ConstPtr((bRow).rawAddress + ((i + j * CurrentCpu.EPR).toLong() * (PrimitiveLayouts.F32).strideBytes))))
        }
        i += CurrentCpu.STEP
    }
    CurrentCpu.vecReduce(sum, c1)
    while (i < k) {
        (c1).let { targetPtr ->
            val targetLayout = PrimitiveLayouts.F32
            var currentValue = targetLayout.readNative(targetPtr.rawAddress)
            currentValue += ((PrimitiveLayouts.F32).readNative((ConstPtr((aRow).rawAddress + ((i).toLong() * (PrimitiveLayouts.F32).strideBytes))).rawAddress) * (PrimitiveLayouts.F32).readNative((ConstPtr((bRow).rawAddress + ((i).toLong() * (PrimitiveLayouts.F32).strideBytes))).rawAddress))
            targetLayout.writeNative(targetPtr.rawAddress, currentValue)
        }
        i += 1
    }
}

internal fun vecSum(row: ConstPtr, b: MutPtr, k: Int) {
    val b1 = b
    val np = k and (CurrentCpu.STEP - 1).inv()
    val sum = CurrentCpu.zeroArray()
    val x = CurrentCpu.zeroArray()
    for (i in (0 until np).stepBy(CurrentCpu.STEP)) {
        for (j in 0 until CurrentCpu.ARR) {
            x[j] = CurrentCpu.load(ConstPtr((row).rawAddress + ((i + j * CurrentCpu.EPR).toLong() * (PrimitiveLayouts.F32).strideBytes)))
            sum[j] = CurrentCpu.vecAdd(sum[j], x[j])
        }
    }
    CurrentCpu.vecReduce(sum, b1)
    for (i1 in np until k) {
        (b1).let { targetPtr ->
            val targetLayout = PrimitiveLayouts.F32
            var currentValue = targetLayout.readNative(targetPtr.rawAddress)
            currentValue += ((PrimitiveLayouts.F32).readNative((ConstPtr((row).rawAddress + ((i1).toLong() * (PrimitiveLayouts.F32).strideBytes))).rawAddress))
            targetLayout.writeNative(targetPtr.rawAddress, currentValue)
        }
    }
}

internal fun vecAddF16(aRow: ConstPtr, bRow: ConstPtr, c: MutPtr, k: Int) {
    var i = 0
    while (i + CurrentCpuF16.STEP <= k) {
        for (j in 0 until CurrentCpuF16.ARR) {
            CurrentCpuF16.vecStore(MutPtr((c).rawAddress + ((i + j * CurrentCpuF16.EPR).toLong() * (ObjectSlotLayout<F16>()).strideBytes)), CurrentCpuF16.vecAdd(CurrentCpuF16.load(ConstPtr((aRow).rawAddress + ((i + j * CurrentCpuF16.EPR).toLong() * (ObjectSlotLayout<F16>()).strideBytes))), CurrentCpuF16.load(ConstPtr((bRow).rawAddress + ((i + j * CurrentCpuF16.EPR).toLong() * (ObjectSlotLayout<F16>()).strideBytes)))))
        }
        i += CurrentCpuF16.STEP
    }
    for (j1 in i until k) {
        (ObjectSlotLayout<F16>()).writeNative((MutPtr((c).rawAddress + ((j1).toLong() * (ObjectSlotLayout<F16>()).strideBytes))).rawAddress, (ObjectSlotLayout<F16>()).readNative((ConstPtr((aRow).rawAddress + ((j1).toLong() * (ObjectSlotLayout<F16>()).strideBytes))).rawAddress) + (ObjectSlotLayout<F16>()).readNative((ConstPtr((bRow).rawAddress + ((j1).toLong() * (ObjectSlotLayout<F16>()).strideBytes))).rawAddress))
    }
}

internal fun vecAddBf16(aRow: ConstPtr, bRow: ConstPtr, c: MutPtr, k: Int) {
    var i = 0
    while (i + CurrentCpuBF16.STEP <= k) {
        for (j in 0 until CurrentCpuBF16.ARR) {
            CurrentCpuBF16.vecStore(MutPtr((c).rawAddress + ((i + j * CurrentCpuBF16.EPR).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes)), CurrentCpuBF16.vecAdd(CurrentCpuBF16.load(ConstPtr((aRow).rawAddress + ((i + j * CurrentCpuBF16.EPR).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))), CurrentCpuBF16.load(ConstPtr((bRow).rawAddress + ((i + j * CurrentCpuBF16.EPR).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes)))))
        }
        i += CurrentCpuBF16.STEP
    }
    for (j1 in i until k) {
        (ObjectSlotLayout<Bf16>()).writeNative((MutPtr((c).rawAddress + ((j1).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress, (ObjectSlotLayout<Bf16>()).readNative((ConstPtr((aRow).rawAddress + ((j1).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress) + (ObjectSlotLayout<Bf16>()).readNative((ConstPtr((bRow).rawAddress + ((j1).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress))
    }
}

internal fun vecScalarAddF16(scalar: F16, xs: ConstPtr, ys: MutPtr, k: Int) {
    val sv = CurrentCpuF16.fromF32(scalar.toF32())
    var i = 0
    while (i + CurrentCpuF16.STEP <= k) {
        for (j in 0 until CurrentCpuF16.ARR) {
            CurrentCpuF16.vecStore(MutPtr((ys).rawAddress + ((i + j * CurrentCpuF16.EPR).toLong() * (ObjectSlotLayout<F16>()).strideBytes)), CurrentCpuF16.vecAdd(CurrentCpuF16.load(ConstPtr((xs).rawAddress + ((i + j * CurrentCpuF16.EPR).toLong() * (ObjectSlotLayout<F16>()).strideBytes))), sv))
        }
        i += CurrentCpuF16.STEP
    }
    for (j1 in i until k) {
        (ObjectSlotLayout<F16>()).writeNative((MutPtr((ys).rawAddress + ((j1).toLong() * (ObjectSlotLayout<F16>()).strideBytes))).rawAddress, (ObjectSlotLayout<F16>()).readNative((ConstPtr((xs).rawAddress + ((j1).toLong() * (ObjectSlotLayout<F16>()).strideBytes))).rawAddress) + scalar)
    }
}

internal fun vecScalarAddBf16(scalar: Bf16, xs: ConstPtr, ys: MutPtr, k: Int) {
    val sv = CurrentCpuBF16.fromF32(scalar.toF32())
    var i = 0
    while (i + CurrentCpuBF16.STEP <= k) {
        for (j in 0 until CurrentCpuBF16.ARR) {
            CurrentCpuBF16.vecStore(MutPtr((ys).rawAddress + ((i + j * CurrentCpuBF16.EPR).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes)), CurrentCpuBF16.vecAdd(CurrentCpuBF16.load(ConstPtr((xs).rawAddress + ((i + j * CurrentCpuBF16.EPR).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))), sv))
        }
        i += CurrentCpuBF16.STEP
    }
    for (j1 in i until k) {
        (ObjectSlotLayout<Bf16>()).writeNative((MutPtr((ys).rawAddress + ((j1).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress, (ObjectSlotLayout<Bf16>()).readNative((ConstPtr((xs).rawAddress + ((j1).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress) + scalar)
    }
}
