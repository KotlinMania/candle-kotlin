// port-lint: source cpu/neon.rs
package io.github.kotlinmania.candle.cpu.neon

import io.github.kotlinmania.candle.cpu.Cpu
import io.github.kotlinmania.candle.cpu.CpuBF16
import io.github.kotlinmania.candle.cpu.CpuF16
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.rustlang.core.arch.aarch64.*
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

        public fun reduceOne(x: Float32x4T): Float = vaddvqF32(x)

        public val STEP: Int = 32

        public val EPR: Int = 4

        public fun zero(): Float32x4T = vdupqNF32(0.0)

        public fun fromF32(x: Float): Float32x4T = vdupqNF32(x)

        public fun zeroArray(): List<Float32x4T> = List((CurrentCpu.ARR).toInt()) { CurrentCpu.zero() }

        public fun load(memAddr: ConstPtr): Float32x4T = vld1qF32(memAddr)

        public fun vecAdd(a: Float32x4T, b: Float32x4T): Float32x4T = vaddqF32(a, b)

        public fun vecFma(a: Float32x4T, b: Float32x4T, c: Float32x4T): Float32x4T = vfmaqF32(a, b, c)

        public fun vecStore(memAddr: MutPtr, a: Float32x4T) {
            vst1qF32(memAddr, a)
        }

        public fun vecReduce(x: List<Float32x4T>, y: MutPtr) {
            val y1 = y
            for (i in 0 until CurrentCpu.ARR / 2) {
                x[2 * i] = vaddqF32(x[2 * i], x[2 * i + 1])
            }
            for (i1 in 0 until CurrentCpu.ARR / 4) {
                x[4 * i1] = vaddqF32(x[4 * i1], x[4 * i1 + 2])
            }
            for (i2 in 0 until CurrentCpu.ARR / 8) {
                x[8 * i2] = vaddqF32(x[8 * i2], x[8 * i2 + 4])
            }
            (PrimitiveLayouts.F32).writeNative((y1).rawAddress, CurrentCpu.reduceOne(x[0]))
        }
    }
}

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

        public val STEP: Int = 32

        public val EPR: Int = 4

        public fun zero(): Float32x4T = vdupqNF32(0.0)

        public fun zeroArray(): List<Float32x4T> = List((CurrentCpuF16.ARR).toInt()) { CurrentCpuF16.zero() }

        public fun load(memAddr: ConstPtr): Float32x4T {
            val bits = vld1U16(ConstPtr(memAddr.rawAddress, memAddr.rawMeta))
            val result: Float32x4T
            asm("fcvtl {dst:v}.4s, {src:v}.4h", src, vreg, bits, dst, out, vreg, result, options(pure, nomem, nostack))
            return result
        }

        public fun vecAdd(a: Float32x4T, b: Float32x4T): Float32x4T = vaddqF32(a, b)

        public fun vecFma(a: Float32x4T, b: Float32x4T, c: Float32x4T): Float32x4T = vfmaqF32(a, b, c)

        public fun vecReduce(x: List<Float32x4T>, y: MutPtr) {
            val y1 = y
            for (i in 0 until CurrentCpuF16.ARR / 2) {
                x[2 * i] = vaddqF32(x[2 * i], x[2 * i + 1])
            }
            for (i1 in 0 until CurrentCpuF16.ARR / 4) {
                x[4 * i1] = vaddqF32(x[4 * i1], x[4 * i1 + 2])
            }
            for (i2 in 0 until CurrentCpuF16.ARR / 8) {
                x[8 * i2] = vaddqF32(x[8 * i2], x[8 * i2 + 4])
            }
            (PrimitiveLayouts.F32).writeNative((y1).rawAddress, vaddvqF32(x[0]))
        }

        public fun fromF32(v: Float): Float32x4T = vdupqNF32(v)

        public fun vecStore(memAddr: MutPtr, a: Float32x4T) {
            val bits: Uint16x4T
            asm("fcvtn {dst:v}.4h, {src:v}.4s", src, vreg, a, dst, out, vreg, bits, options(pure, nomem, nostack))
            vst1U16(MutPtr(memAddr.rawAddress, memAddr.rawMeta), bits)
        }
    }
}

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

        public val STEP: Int = 32

        public val EPR: Int = 4

        public fun zero(): Float32x4T = vdupqNF32(0.0)

        public fun zeroArray(): List<Float32x4T> = List((CurrentCpuBF16.ARR).toInt()) { CurrentCpuBF16.zero() }

        public fun load(memAddr: ConstPtr): Float32x4T {
            // bf16 is the top 16 bits of f32; shift left by 16 to reconstruct f32 bits.
            val i = vld1U16(ConstPtr(memAddr.rawAddress, memAddr.rawMeta))
            val s = vshllNU16(i, 16)
            return vreinterpretqF32U32(s)
        }

        public fun vecAdd(a: Float32x4T, b: Float32x4T): Float32x4T = vaddqF32(a, b)

        public fun vecFma(a: Float32x4T, b: Float32x4T, c: Float32x4T): Float32x4T = vfmaqF32(a, b, c)

        public fun vecReduce(x: List<Float32x4T>, y: MutPtr) {
            val y1 = y
            for (i in 0 until CurrentCpuBF16.ARR / 2) {
                x[2 * i] = vaddqF32(x[2 * i], x[2 * i + 1])
            }
            for (i1 in 0 until CurrentCpuBF16.ARR / 4) {
                x[4 * i1] = vaddqF32(x[4 * i1], x[4 * i1 + 2])
            }
            for (i2 in 0 until CurrentCpuBF16.ARR / 8) {
                x[8 * i2] = vaddqF32(x[8 * i2], x[8 * i2 + 4])
            }
            (PrimitiveLayouts.F32).writeNative((y1).rawAddress, vaddvqF32(x[0]))
        }

        public fun fromF32(v: Float): Float32x4T = vdupqNF32(v)

        public fun vecStore(memAddr: MutPtr, a: Float32x4T) {
            // Extract top 16 bits of each f32 lane, which is the bf16 representation.
            val s = vreinterpretqU32F32(a)
            val h = vshrnNU32(s, 16)
            vst1U16(MutPtr(memAddr.rawAddress, memAddr.rawMeta), h)
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

internal fun vecDotF16(aRow: ConstPtr, bRow: ConstPtr, c: MutPtr, k: Int) {
    val c1 = c
    var sumf = 0.0f
    var sum = CurrentCpuF16.zeroArray()
    var stepsSinceFlush = 0
    var i = 0
    while (i + CurrentCpuF16.STEP <= k) {
        for (j in 0 until CurrentCpuF16.ARR) {
            sum[j] = CurrentCpuF16.vecFma(sum[j], CurrentCpuF16.load(ConstPtr((aRow).rawAddress + ((i + j * CurrentCpuF16.EPR).toLong() * (ObjectSlotLayout<F16>()).strideBytes))), CurrentCpuF16.load(ConstPtr((bRow).rawAddress + ((i + j * CurrentCpuF16.EPR).toLong() * (ObjectSlotLayout<F16>()).strideBytes))))
        }
        stepsSinceFlush += 1
        if (stepsSinceFlush == CurrentCpuF16.FLUSH_INTERVAL) {
            val partial = 0.0f
            CurrentCpuF16.vecReduce(sum, partial)
            sumf += partial
            sum = CurrentCpuF16.zeroArray()
            stepsSinceFlush = 0
        }
        i += CurrentCpuF16.STEP
    }
    val partial1 = 0.0f
    CurrentCpuF16.vecReduce(sum, partial1)
    sumf += partial1
    while (i < k) {
        sumf += (ObjectSlotLayout<F16>()).readNative((ConstPtr((aRow).rawAddress + ((i).toLong() * (ObjectSlotLayout<F16>()).strideBytes))).rawAddress).toF32() * (ObjectSlotLayout<F16>()).readNative((ConstPtr((bRow).rawAddress + ((i).toLong() * (ObjectSlotLayout<F16>()).strideBytes))).rawAddress).toF32()
        i += 1
    }
    (PrimitiveLayouts.F32).writeNative((c1).rawAddress, sumf)
}

internal fun vecDotBf16(aRow: ConstPtr, bRow: ConstPtr, c: MutPtr, k: Int) {
    val c1 = c
    val sum = CurrentCpuBF16.zeroArray()
    var i = 0
    while (i + CurrentCpuBF16.STEP <= k) {
        for (j in 0 until CurrentCpuBF16.ARR) {
            sum[j] = CurrentCpuBF16.vecFma(sum[j], CurrentCpuBF16.load(ConstPtr((aRow).rawAddress + ((i + j * CurrentCpuBF16.EPR).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))), CurrentCpuBF16.load(ConstPtr((bRow).rawAddress + ((i + j * CurrentCpuBF16.EPR).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))))
        }
        i += CurrentCpuBF16.STEP
    }
    CurrentCpuBF16.vecReduce(sum, c1)
    while (i < k) {
        (c1).let { targetPtr ->
            val targetLayout = PrimitiveLayouts.F32
            var currentValue = targetLayout.readNative(targetPtr.rawAddress)
            currentValue += ((ObjectSlotLayout<Bf16>()).readNative((ConstPtr((aRow).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress).toF32() * (ObjectSlotLayout<Bf16>()).readNative((ConstPtr((bRow).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress).toF32())
            targetLayout.writeNative(targetPtr.rawAddress, currentValue)
        }
        i += 1
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
    // Widen to f32, add, narrow. The bfdot-specialized CurrentCpuBF16 keeps raw bf16 bits in
    // its Unit and its vecAdd interprets them as f32 lanes, so it must not be used here.
    var i = 0
    while (i + 8 <= k) {
        val a = vld1qU16(ConstPtr(ConstPtr((aRow).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes)).rawAddress, ConstPtr((aRow).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes)).rawMeta))
        val b = vld1qU16(ConstPtr(ConstPtr((bRow).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes)).rawAddress, ConstPtr((bRow).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes)).rawMeta))
        val aLo = vreinterpretqF32U32(vshllNU16(vgetLowU16(a), 16))
        val aHi = vreinterpretqF32U32(vshllHighNU16(a, 16))
        val bLo = vreinterpretqF32U32(vshllNU16(vgetLowU16(b), 16))
        val bHi = vreinterpretqF32U32(vshllHighNU16(b, 16))
        val lo = vshrnNU32(vreinterpretqU32F32(vaddqF32(aLo, bLo)), 16)
        val hi = vshrnNU32(vreinterpretqU32F32(vaddqF32(aHi, bHi)), 16)
        vst1qU16(MutPtr(MutPtr((c).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes)).rawAddress, MutPtr((c).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes)).rawMeta), vcombineU16(lo, hi))
        i += 8
    }
    for (j in i until k) {
        (ObjectSlotLayout<Bf16>()).writeNative((MutPtr((c).rawAddress + ((j).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress, (ObjectSlotLayout<Bf16>()).readNative((ConstPtr((aRow).rawAddress + ((j).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress) + (ObjectSlotLayout<Bf16>()).readNative((ConstPtr((bRow).rawAddress + ((j).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress))
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

internal fun vecMulBf16(aRow: ConstPtr, bRow: ConstPtr, c: MutPtr, k: Int) {
    var i = 0
    while (i + 8 <= k) {
        val a = vld1qU16(ConstPtr(ConstPtr((aRow).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes)).rawAddress, ConstPtr((aRow).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes)).rawMeta))
        val b = vld1qU16(ConstPtr(ConstPtr((bRow).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes)).rawAddress, ConstPtr((bRow).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes)).rawMeta))
        val aLo = vreinterpretqF32U32(vshllNU16(vgetLowU16(a), 16))
        val aHi = vreinterpretqF32U32(vshllHighNU16(a, 16))
        val bLo = vreinterpretqF32U32(vshllNU16(vgetLowU16(b), 16))
        val bHi = vreinterpretqF32U32(vshllHighNU16(b, 16))
        val lo = vshrnNU32(vreinterpretqU32F32(vmulqF32(aLo, bLo)), 16)
        val hi = vshrnNU32(vreinterpretqU32F32(vmulqF32(aHi, bHi)), 16)
        vst1qU16(MutPtr(MutPtr((c).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes)).rawAddress, MutPtr((c).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes)).rawMeta), vcombineU16(lo, hi))
        i += 8
    }
    for (j in i until k) {
        (ObjectSlotLayout<Bf16>()).writeNative((MutPtr((c).rawAddress + ((j).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress, (ObjectSlotLayout<Bf16>()).readNative((ConstPtr((aRow).rawAddress + ((j).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress) * (ObjectSlotLayout<Bf16>()).readNative((ConstPtr((bRow).rawAddress + ((j).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress))
    }
}

internal fun vecScalarAddBf16(scalar: Bf16, xs: ConstPtr, ys: MutPtr, k: Int) {
    val sv = vdupqNF32(scalar.toF32())
    var i = 0
    while (i + 8 <= k) {
        val x = vld1qU16(ConstPtr(ConstPtr((xs).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes)).rawAddress, ConstPtr((xs).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes)).rawMeta))
        val xLo = vreinterpretqF32U32(vshllNU16(vgetLowU16(x), 16))
        val xHi = vreinterpretqF32U32(vshllHighNU16(x, 16))
        val lo = vshrnNU32(vreinterpretqU32F32(vaddqF32(xLo, sv)), 16)
        val hi = vshrnNU32(vreinterpretqU32F32(vaddqF32(xHi, sv)), 16)
        vst1qU16(MutPtr(MutPtr((ys).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes)).rawAddress, MutPtr((ys).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes)).rawMeta), vcombineU16(lo, hi))
        i += 8
    }
    for (j in i until k) {
        (ObjectSlotLayout<Bf16>()).writeNative((MutPtr((ys).rawAddress + ((j).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress, (ObjectSlotLayout<Bf16>()).readNative((ConstPtr((xs).rawAddress + ((j).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress) + scalar)
    }
}
