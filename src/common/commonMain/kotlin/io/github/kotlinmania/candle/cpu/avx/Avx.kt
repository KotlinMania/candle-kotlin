// port-lint: source cpu/avx.rs
package io.github.kotlinmania.candle.cpu.avx

import io.github.kotlinmania.candle.cpu.Cpu
import io.github.kotlinmania.candle.cpu.CpuBF16
import io.github.kotlinmania.candle.cpu.CpuF16
import io.github.kotlinmania.half.Bf16
import io.github.kotlinmania.half.F16
import io.github.kotlinmania.rustlang.core.arch.x8664.*
import io.github.kotlinmania.rustlang.core.mem.ObjectSlotLayout
import io.github.kotlinmania.rustlang.core.mem.PrimitiveLayouts
import io.github.kotlinmania.rustlang.core.mem.TypeLayout
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr
import io.github.kotlinmania.rustlang.std.mem.transmute

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

        public fun zero(): M256 = mm256SetzeroPs()

        public fun zeroArray(): List<M256> = List(ARR) { CurrentCpu.zero() }

        public fun fromF32(v: Float): M256 = mm256Set1Ps(v)

        public fun load(memAddr: ConstPtr): M256 = mm256LoaduPs(memAddr)

        public fun vecAdd(a: M256, b: M256): M256 = mm256AddPs(a, b)

        public fun vecFma(a: M256, b: M256, c: M256): M256 = mm256AddPs(mm256MulPs(b, c), a)

        public fun vecStore(memAddr: MutPtr, a: M256) {
            mm256StoreuPs(memAddr, a)
        }

        public fun vecReduce(x: List<M256>, y: MutPtr) {
            val y1 = y
            for (i in 0 until ARR / 2) {
                x[2 * i] = mm256AddPs(x[2 * i], x[2 * i + 1])
            }
            for (i1 in 0 until ARR / 4) {
                x[4 * i1] = mm256AddPs(x[4 * i1], x[4 * i1 + 2])
            }
            for (i2 in 0 until ARR / 8) {
                x[8 * i2] = mm256AddPs(x[8 * i2], x[8 * i2 + 4])
            }
            val t0 = mmAddPs(mm256Castps256Ps128(x[0]), mm256Extractf128Ps(x[0], 1))
            val t1 = mmHaddPs(t0, t0)
            (PrimitiveLayouts.F32).writeNative((y1).rawAddress, mmCvtssF32(mmHaddPs(t1, t1)))
        }
    }
}

public val STEP: Int = 32

public val EPR: Int = 8

public val ARR: Int = STEP / EPR

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

        public fun zero(): M256 = mm256SetzeroPs()

        public fun zeroArray(): List<M256> = List(ARR) { CurrentCpuF16.zero() }

        public fun fromF32(v: Float): M256 = mm256Set1Ps(v)

        public fun load(memAddr: ConstPtr): M256 = mm256CvtphPs(mmLoaduSi128(ConstPtr(memAddr.rawAddress, memAddr.rawMeta)))

        public fun loadCpuF16(memAddr: ConstPtr): M256 {
            val tmp = List(8) { 0.0f }
            for (i in 0 until 8) {
                tmp[i] = (ObjectSlotLayout<F16>()).readNative((ConstPtr((memAddr).rawAddress + ((i).toLong() * (ObjectSlotLayout<F16>()).strideBytes))).rawAddress).toF32()
            }
            return mm256LoaduPs(tmp.asPtr())
        }

        public fun vecAdd(a: M256, b: M256): M256 = mm256AddPs(a, b)

        public fun vecFma(a: M256, b: M256, c: M256): M256 = mm256AddPs(mm256MulPs(b, c), a)

        public fun vecStore(memAddr: MutPtr, a: M256) {
            mmStoreuSi128(MutPtr(memAddr.rawAddress, memAddr.rawMeta), mm256CvtpsPh(a, 0))
        }

        public fun vecStoreCpuF16(memAddr: MutPtr, a: M256) {
            val tmp = List(8) { 0.0f }
            mm256StoreuPs(tmp.asMutPtr(), a)
            for (i in 0 until 8) {
                (ObjectSlotLayout<F16>()).writeNative((MutPtr((memAddr).rawAddress + ((i).toLong() * (ObjectSlotLayout<F16>()).strideBytes))).rawAddress, fromF32(tmp[i]))
            }
        }

        public fun vecReduce(x: List<M256>, y: MutPtr) {
            val y1 = y
            var offset = ARR shr 1
            for (i in 0 until offset) {
                x[i] = mm256AddPs(x[i], x[offset + i])
            }
            offset = offset shr 1
            for (i1 in 0 until offset) {
                x[i1] = mm256AddPs(x[i1], x[offset + i1])
            }
            offset = offset shr 1
            for (i2 in 0 until offset) {
                x[i2] = mm256AddPs(x[i2], x[offset + i2])
            }
            val t0 = mmAddPs(mm256Castps256Ps128(x[0]), mm256Extractf128Ps(x[0], 1))
            val t1 = mmHaddPs(t0, t0)
            (PrimitiveLayouts.F32).writeNative((y1).rawAddress, mmCvtssF32(mmHaddPs(t1, t1)))
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

        public val STEP: Int = STEP

        public val EPR: Int = EPR

        public val ARR: Int = ARR

        public fun zero(): M256 = mm256SetzeroPs()

        public fun zeroArray(): List<M256> = List(ARR) { CurrentCpuBF16.zero() }

        public fun fromF32(v: Float): M256 = mm256Set1Ps(v)

        public fun load(memAddr: ConstPtr): M256 =
            if (isX86FeatureDetected("avx512bf16") && isX86FeatureDetected("avx512vl")) {
                mm256CvtpbhPs(transmute<M128i, M128bh>(mmLoaduSi128(ConstPtr(memAddr.rawAddress, memAddr.rawMeta))))
            } else {
                val tmp = List(8) { 0.0f }
                for (i in 0 until 8) {
                    tmp[i] = (ObjectSlotLayout<Bf16>()).readNative((ConstPtr((memAddr).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress).toF32()
                }
                mm256LoaduPs(tmp.asPtr())
            }

        public fun vecAdd(a: M256, b: M256): M256 = mm256AddPs(a, b)

        public fun vecFma(a: M256, b: M256, c: M256): M256 = mm256AddPs(mm256MulPs(b, c), a)

        public fun vecStore(memAddr: MutPtr, a: M256) {
            if (isX86FeatureDetected("avx512bf16") && isX86FeatureDetected("avx512vl")) {
                mmStoreuSi128(MutPtr(memAddr.rawAddress, memAddr.rawMeta), transmute<M128bh, M128i>(mm256CvtnepsPbh(a)))
            } else {
                val tmp = List(8) { 0.0f }
                mm256StoreuPs(tmp.asMutPtr(), a)
                for (i in 0 until 8) {
                    (ObjectSlotLayout<Bf16>()).writeNative((MutPtr((memAddr).rawAddress + ((i).toLong() * (ObjectSlotLayout<Bf16>()).strideBytes))).rawAddress, fromF32(tmp[i]))
                }
            }
        }

        public fun vecReduce(x: List<M256>, y: MutPtr) {
            val y1 = y
            var offset = ARR shr 1
            for (i in 0 until offset) {
                x[i] = mm256AddPs(x[i], x[offset + i])
            }
            offset = offset shr 1
            for (i1 in 0 until offset) {
                x[i1] = mm256AddPs(x[i1], x[offset + i1])
            }
            offset = offset shr 1
            for (i2 in 0 until offset) {
                x[i2] = mm256AddPs(x[i2], x[offset + i2])
            }
            val t0 = mmAddPs(mm256Castps256Ps128(x[0]), mm256Extractf128Ps(x[0], 1))
            val t1 = mmHaddPs(t0, t0)
            (PrimitiveLayouts.F32).writeNative((y1).rawAddress, mmCvtssF32(mmHaddPs(t1, t1)))
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
