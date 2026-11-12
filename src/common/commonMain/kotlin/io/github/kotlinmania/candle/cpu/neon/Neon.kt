// port-lint: source cpu/neon.rs
package io.github.kotlinmania.candle.cpu.neon

import io.github.kotlinmania.candle.COpaquePointer
import io.github.kotlinmania.candle.cpu.Cpu
import io.github.kotlinmania.candle.cpu.CpuBF16
import io.github.kotlinmania.candle.cpu.CpuF16
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16

public open class CurrentCpu : Cpu {
    public companion object {
        public fun reduceOne(x: Float32x4T): Float = vaddvqF32(x)

        public val STEP: Int = 32

        public val EPR: Int = 4

        public fun zero(): Float32x4T = vdupqNF32(0.0)

        public fun fromF32(x: Float): Float32x4T = vdupqNF32(x)

        public fun zeroArray(): List<Float32x4T> = List((CurrentCpu.ARR).toInt()) { CurrentCpu.zero() }

        public fun load(memAddr: COpaquePointer?): Float32x4T = vld1qF32(memAddr)

        public fun vecAdd(a: Float32x4T, b: Float32x4T): Float32x4T = vaddqF32(a, b)

        public fun vecFma(a: Float32x4T, b: Float32x4T, c: Float32x4T): Float32x4T = vfmaqF32(a, b, c)

        public fun vecStore(memAddr: COpaquePointer?, a: Float32x4T) {
            vst1qF32(memAddr, a)
        }

        public fun vecReduce(x: List<Float32x4T>, y: org.rustlang.runtime.RustPointer<Float>?) {
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
            (y1)!!.write(CurrentCpu.reduceOne(x[0]))
        }
    }
}

public open class CurrentCpuF16 : CpuF16 {
    public companion object {
        public val STEP: Int = 32

        public val EPR: Int = 4

        public fun zero(): Float32x4T = vdupqNF32(0.0)

        public fun zeroArray(): List<Float32x4T> = List((CurrentCpuF16.ARR).toInt()) { CurrentCpuF16.zero() }

        public fun load(memAddr: org.rustlang.runtime.RustPointer<F16>?): Float32x4T {
            val bits = vld1U16((memAddr)?.reinterpret(org.rustlang.runtime.AtomicScalarCodec.U16))
            val result: Float32x4T
            asm("fcvtl {dst:v}.4s, {src:v}.4h", src, vreg, bits, dst, out, vreg, result, options(pure, nomem, nostack))
            return result
        }

        public fun vecAdd(a: Float32x4T, b: Float32x4T): Float32x4T = vaddqF32(a, b)

        public fun vecFma(a: Float32x4T, b: Float32x4T, c: Float32x4T): Float32x4T = vfmaqF32(a, b, c)

        public fun vecReduce(x: List<Float32x4T>, y: org.rustlang.runtime.RustPointer<Float>?) {
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
            (y1)!!.write(vaddvqF32(x[0]))
        }

        public fun fromF32(v: Float): Float32x4T = vdupqNF32(v)

        public fun vecStore(memAddr: org.rustlang.runtime.RustPointer<F16>?, a: Float32x4T) {
            val bits: Uint16x4T
            asm("fcvtn {dst:v}.4h, {src:v}.4s", src, vreg, a, dst, out, vreg, bits, options(pure, nomem, nostack))
            vst1U16((memAddr)?.reinterpret(org.rustlang.runtime.AtomicScalarCodec.U16), bits)
        }
    }
}

public open class CurrentCpuBF16 : CpuBF16 {
    public companion object {
        public val STEP: Int = 32

        public val EPR: Int = 4

        public fun zero(): Float32x4T = vdupqNF32(0.0)

        public fun zeroArray(): List<Float32x4T> = List((CurrentCpuBF16.ARR).toInt()) { CurrentCpuBF16.zero() }

        public fun load(memAddr: org.rustlang.runtime.RustPointer<Bf16>?): Float32x4T {
            // bf16 is the top 16 bits of f32; shift left by 16 to reconstruct f32 bits.
            val i = vld1U16((memAddr)?.reinterpret(org.rustlang.runtime.AtomicScalarCodec.U16))
            val s = vshllNU16(i, 16)
            return vreinterpretqF32U32(s)
        }

        public fun vecAdd(a: Float32x4T, b: Float32x4T): Float32x4T = vaddqF32(a, b)

        public fun vecFma(a: Float32x4T, b: Float32x4T, c: Float32x4T): Float32x4T = vfmaqF32(a, b, c)

        public fun vecReduce(x: List<Float32x4T>, y: org.rustlang.runtime.RustPointer<Float>?) {
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
            (y1)!!.write(vaddvqF32(x[0]))
        }

        public fun fromF32(v: Float): Float32x4T = vdupqNF32(v)

        public fun vecStore(memAddr: org.rustlang.runtime.RustPointer<Bf16>?, a: Float32x4T) {
            // Extract top 16 bits of each f32 lane, which is the bf16 representation.
            val s = vreinterpretqU32F32(a)
            val h = vshrnNU32(s, 16)
            vst1U16((memAddr)?.reinterpret(org.rustlang.runtime.AtomicScalarCodec.U16), h)
        }
    }
}

internal fun vecDotF32(aRow: org.rustlang.runtime.RustPointer<Float>?, bRow: org.rustlang.runtime.RustPointer<Float>?, c: org.rustlang.runtime.RustPointer<Float>?, k: Int) {
    val c1 = c
    val sum = CurrentCpu.zeroArray()
    var i = 0
    while (i + CurrentCpu.STEP <= k) {
        for (j in 0 until CurrentCpu.ARR) {
            sum[j] = CurrentCpu.vecFma(sum[j], CurrentCpu.load(org.rustlang.runtime.rustPointerOffset(aRow, null, (i + j * CurrentCpu.EPR).toLong(), false, false)), CurrentCpu.load(org.rustlang.runtime.rustPointerOffset(bRow, null, (i + j * CurrentCpu.EPR).toLong(), false, false)))
        }
        i += CurrentCpu.STEP
    }
    CurrentCpu.vecReduce(sum, c1)
    while (i < k) {
        (c1)!!.let { __rustPlace -> var __rustValue = __rustPlace.read(); __rustValue += ((org.rustlang.runtime.rustPointerOffset(aRow, null, (i).toLong(), false, false))!!.read() * (org.rustlang.runtime.rustPointerOffset(bRow, null, (i).toLong(), false, false))!!.read()); __rustPlace.write(__rustValue) }
        i += 1
    }
}

internal fun vecSum(row: org.rustlang.runtime.RustPointer<Float>?, b: org.rustlang.runtime.RustPointer<Float>?, k: Int) {
    val b1 = b
    val np = k and (CurrentCpu.STEP - 1).inv()
    val sum = CurrentCpu.zeroArray()
    val x = CurrentCpu.zeroArray()
    for (i in (0 until np).stepBy(CurrentCpu.STEP)) {
        for (j in 0 until CurrentCpu.ARR) {
            x[j] = CurrentCpu.load(org.rustlang.runtime.rustPointerOffset(row, null, (i + j * CurrentCpu.EPR).toLong(), false, false))
            sum[j] = CurrentCpu.vecAdd(sum[j], x[j])
        }
    }
    CurrentCpu.vecReduce(sum, b1)
    for (i1 in np until k) {
        (b1)!!.let { __rustPlace -> var __rustValue = __rustPlace.read(); __rustValue += ((org.rustlang.runtime.rustPointerOffset(row, null, (i1).toLong(), false, false))!!.read()); __rustPlace.write(__rustValue) }
    }
}

internal fun vecDotF16(aRow: org.rustlang.runtime.RustPointer<F16>?, bRow: org.rustlang.runtime.RustPointer<F16>?, c: org.rustlang.runtime.RustPointer<Float>?, k: Int) {
    val c1 = c
    var sumf = 0.0f
    var sum = CurrentCpuF16.zeroArray()
    var stepsSinceFlush = 0
    var i = 0
    while (i + CurrentCpuF16.STEP <= k) {
        for (j in 0 until CurrentCpuF16.ARR) {
            sum[j] = CurrentCpuF16.vecFma(sum[j], CurrentCpuF16.load(org.rustlang.runtime.rustPointerOffset(aRow, null, (i + j * CurrentCpuF16.EPR).toLong(), false, false)), CurrentCpuF16.load(org.rustlang.runtime.rustPointerOffset(bRow, null, (i + j * CurrentCpuF16.EPR).toLong(), false, false)))
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
        sumf += (org.rustlang.runtime.rustPointerOffset(aRow, null, (i).toLong(), false, false))!!.read().toF32() * (org.rustlang.runtime.rustPointerOffset(bRow, null, (i).toLong(), false, false))!!.read().toF32()
        i += 1
    }
    (c1)!!.write(sumf)
}

internal fun vecDotBf16(aRow: org.rustlang.runtime.RustPointer<Bf16>?, bRow: org.rustlang.runtime.RustPointer<Bf16>?, c: org.rustlang.runtime.RustPointer<Float>?, k: Int) {
    val c1 = c
    val sum = CurrentCpuBF16.zeroArray()
    var i = 0
    while (i + CurrentCpuBF16.STEP <= k) {
        for (j in 0 until CurrentCpuBF16.ARR) {
            sum[j] = CurrentCpuBF16.vecFma(sum[j], CurrentCpuBF16.load(org.rustlang.runtime.rustPointerOffset(aRow, null, (i + j * CurrentCpuBF16.EPR).toLong(), false, false)), CurrentCpuBF16.load(org.rustlang.runtime.rustPointerOffset(bRow, null, (i + j * CurrentCpuBF16.EPR).toLong(), false, false)))
        }
        i += CurrentCpuBF16.STEP
    }
    CurrentCpuBF16.vecReduce(sum, c1)
    while (i < k) {
        (c1)!!.let { __rustPlace -> var __rustValue = __rustPlace.read(); __rustValue += ((org.rustlang.runtime.rustPointerOffset(aRow, null, (i).toLong(), false, false))!!.read().toF32() * (org.rustlang.runtime.rustPointerOffset(bRow, null, (i).toLong(), false, false))!!.read().toF32()); __rustPlace.write(__rustValue) }
        i += 1
    }
}

internal fun vecAddF16(aRow: org.rustlang.runtime.RustPointer<F16>?, bRow: org.rustlang.runtime.RustPointer<F16>?, c: org.rustlang.runtime.RustPointer<F16>?, k: Int) {
    var i = 0
    while (i + CurrentCpuF16.STEP <= k) {
        for (j in 0 until CurrentCpuF16.ARR) {
            CurrentCpuF16.vecStore(org.rustlang.runtime.rustPointerOffset(c, null, (i + j * CurrentCpuF16.EPR).toLong(), false, false), CurrentCpuF16.vecAdd(CurrentCpuF16.load(org.rustlang.runtime.rustPointerOffset(aRow, null, (i + j * CurrentCpuF16.EPR).toLong(), false, false)), CurrentCpuF16.load(org.rustlang.runtime.rustPointerOffset(bRow, null, (i + j * CurrentCpuF16.EPR).toLong(), false, false))))
        }
        i += CurrentCpuF16.STEP
    }
    for (j1 in i until k) {
        (org.rustlang.runtime.rustPointerOffset(c, null, (j1).toLong(), false, false))!!.write((org.rustlang.runtime.rustPointerOffset(aRow, null, (j1).toLong(), false, false))!!.read() + (org.rustlang.runtime.rustPointerOffset(bRow, null, (j1).toLong(), false, false))!!.read())
    }
}

internal fun vecAddBf16(aRow: org.rustlang.runtime.RustPointer<Bf16>?, bRow: org.rustlang.runtime.RustPointer<Bf16>?, c: org.rustlang.runtime.RustPointer<Bf16>?, k: Int) {
    // Widen to f32, add, narrow. The bfdot-specialized CurrentCpuBF16 keeps raw bf16 bits in
    // its Unit and its vecAdd interprets them as f32 lanes, so it must not be used here.
    var i = 0
    while (i + 8 <= k) {
        val a = vld1qU16((org.rustlang.runtime.rustPointerOffset(aRow, null, (i).toLong(), false, false))?.reinterpret(org.rustlang.runtime.AtomicScalarCodec.U16))
        val b = vld1qU16((org.rustlang.runtime.rustPointerOffset(bRow, null, (i).toLong(), false, false))?.reinterpret(org.rustlang.runtime.AtomicScalarCodec.U16))
        val aLo = vreinterpretqF32U32(vshllNU16(vgetLowU16(a), 16))
        val aHi = vreinterpretqF32U32(vshllHighNU16(a, 16))
        val bLo = vreinterpretqF32U32(vshllNU16(vgetLowU16(b), 16))
        val bHi = vreinterpretqF32U32(vshllHighNU16(b, 16))
        val lo = vshrnNU32(vreinterpretqU32F32(vaddqF32(aLo, bLo)), 16)
        val hi = vshrnNU32(vreinterpretqU32F32(vaddqF32(aHi, bHi)), 16)
        vst1qU16((org.rustlang.runtime.rustPointerOffset(c, null, (i).toLong(), false, false))?.reinterpret(org.rustlang.runtime.AtomicScalarCodec.U16), vcombineU16(lo, hi))
        i += 8
    }
    for (j in i until k) {
        (org.rustlang.runtime.rustPointerOffset(c, null, (j).toLong(), false, false))!!.write((org.rustlang.runtime.rustPointerOffset(aRow, null, (j).toLong(), false, false))!!.read() + (org.rustlang.runtime.rustPointerOffset(bRow, null, (j).toLong(), false, false))!!.read())
    }
}

internal fun vecScalarAddF16(scalar: F16, xs: org.rustlang.runtime.RustPointer<F16>?, ys: org.rustlang.runtime.RustPointer<F16>?, k: Int) {
    val sv = CurrentCpuF16.fromF32(scalar.toF32())
    var i = 0
    while (i + CurrentCpuF16.STEP <= k) {
        for (j in 0 until CurrentCpuF16.ARR) {
            CurrentCpuF16.vecStore(org.rustlang.runtime.rustPointerOffset(ys, null, (i + j * CurrentCpuF16.EPR).toLong(), false, false), CurrentCpuF16.vecAdd(CurrentCpuF16.load(org.rustlang.runtime.rustPointerOffset(xs, null, (i + j * CurrentCpuF16.EPR).toLong(), false, false)), sv))
        }
        i += CurrentCpuF16.STEP
    }
    for (j1 in i until k) {
        (org.rustlang.runtime.rustPointerOffset(ys, null, (j1).toLong(), false, false))!!.write((org.rustlang.runtime.rustPointerOffset(xs, null, (j1).toLong(), false, false))!!.read() + scalar)
    }
}

internal fun vecMulBf16(aRow: org.rustlang.runtime.RustPointer<Bf16>?, bRow: org.rustlang.runtime.RustPointer<Bf16>?, c: org.rustlang.runtime.RustPointer<Bf16>?, k: Int) {
    var i = 0
    while (i + 8 <= k) {
        val a = vld1qU16((org.rustlang.runtime.rustPointerOffset(aRow, null, (i).toLong(), false, false))?.reinterpret(org.rustlang.runtime.AtomicScalarCodec.U16))
        val b = vld1qU16((org.rustlang.runtime.rustPointerOffset(bRow, null, (i).toLong(), false, false))?.reinterpret(org.rustlang.runtime.AtomicScalarCodec.U16))
        val aLo = vreinterpretqF32U32(vshllNU16(vgetLowU16(a), 16))
        val aHi = vreinterpretqF32U32(vshllHighNU16(a, 16))
        val bLo = vreinterpretqF32U32(vshllNU16(vgetLowU16(b), 16))
        val bHi = vreinterpretqF32U32(vshllHighNU16(b, 16))
        val lo = vshrnNU32(vreinterpretqU32F32(vmulqF32(aLo, bLo)), 16)
        val hi = vshrnNU32(vreinterpretqU32F32(vmulqF32(aHi, bHi)), 16)
        vst1qU16((org.rustlang.runtime.rustPointerOffset(c, null, (i).toLong(), false, false))?.reinterpret(org.rustlang.runtime.AtomicScalarCodec.U16), vcombineU16(lo, hi))
        i += 8
    }
    for (j in i until k) {
        (org.rustlang.runtime.rustPointerOffset(c, null, (j).toLong(), false, false))!!.write((org.rustlang.runtime.rustPointerOffset(aRow, null, (j).toLong(), false, false))!!.read() * (org.rustlang.runtime.rustPointerOffset(bRow, null, (j).toLong(), false, false))!!.read())
    }
}

internal fun vecScalarAddBf16(scalar: Bf16, xs: org.rustlang.runtime.RustPointer<Bf16>?, ys: org.rustlang.runtime.RustPointer<Bf16>?, k: Int) {
    val sv = vdupqNF32(scalar.toF32())
    var i = 0
    while (i + 8 <= k) {
        val x = vld1qU16((org.rustlang.runtime.rustPointerOffset(xs, null, (i).toLong(), false, false))?.reinterpret(org.rustlang.runtime.AtomicScalarCodec.U16))
        val xLo = vreinterpretqF32U32(vshllNU16(vgetLowU16(x), 16))
        val xHi = vreinterpretqF32U32(vshllHighNU16(x, 16))
        val lo = vshrnNU32(vreinterpretqU32F32(vaddqF32(xLo, sv)), 16)
        val hi = vshrnNU32(vreinterpretqU32F32(vaddqF32(xHi, sv)), 16)
        vst1qU16((org.rustlang.runtime.rustPointerOffset(ys, null, (i).toLong(), false, false))?.reinterpret(org.rustlang.runtime.AtomicScalarCodec.U16), vcombineU16(lo, hi))
        i += 8
    }
    for (j in i until k) {
        (org.rustlang.runtime.rustPointerOffset(ys, null, (j).toLong(), false, false))!!.write((org.rustlang.runtime.rustPointerOffset(xs, null, (j).toLong(), false, false))!!.read() + scalar)
    }
}
