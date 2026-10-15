// port-lint: source cuda_backend/utils.rs
package io.github.kotlinmania.candle.cuda.utils

import io.github.kotlinmania.candle.BF16
import io.github.kotlinmania.candle.CudaDevice
import io.github.kotlinmania.candle.CudaStorageSlice
import io.github.kotlinmania.candle.F16
import io.github.kotlinmania.candle.F8E4M3
import io.github.kotlinmania.candle.Layout
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.cuda.error.CudaError

/** Helper functions to plug cuda kernels in candle. */
public typealias S = CudaStorageSlice

private val <T> T.Ok: (T) -> T get() = { it }
private val <T : Throwable> T.Err: (T) -> T get() = { it }
private val <T : Throwable> T.w: () -> T get() = { this }

private val S.U8.s: List<UByte> get() = data
private val S.U32.s: List<UInt> get() = data
private val S.I16.s: List<Short> get() = data
private val S.I32.s: List<Int> get() = data
private val S.I64.s: List<Long> get() = data
private val S.BF16.s: List<BF16> get() = data
private val S.F16.s: List<F16> get() = data
private val S.F32.s: List<Float> get() = data
private val S.F64.s: List<Double> get() = data
private val S.F8E4M3.s: List<F8E4M3> get() = data

private val S.U8.s1: List<UByte> get() = data
private val S.U32.s1: List<UInt> get() = data
private val S.I16.s1: List<Short> get() = data
private val S.I32.s1: List<Int> get() = data
private val S.I64.s1: List<Long> get() = data
private val S.BF16.s1: List<BF16> get() = data
private val S.F16.s1: List<F16> get() = data
private val S.F32.s1: List<Float> get() = data
private val S.F64.s1: List<Double> get() = data
private val S.F8E4M3.s1: List<F8E4M3> get() = data

private val S.U8.s2: List<UByte> get() = data
private val S.U32.s2: List<UInt> get() = data
private val S.I16.s2: List<Short> get() = data
private val S.I32.s2: List<Int> get() = data
private val S.I64.s2: List<Long> get() = data
private val S.BF16.s2: List<BF16> get() = data
private val S.F16.s2: List<F16> get() = data
private val S.F32.s2: List<Float> get() = data
private val S.F64.s2: List<Double> get() = data
private val S.F8E4M3.s2: List<F8E4M3> get() = data

private val S.U8.s3: List<UByte> get() = data
private val S.U32.s3: List<UInt> get() = data
private val S.I64.s3: List<Long> get() = data
private val S.BF16.s3: List<BF16> get() = data
private val S.F16.s3: List<F16> get() = data
private val S.F32.s3: List<Float> get() = data
private val S.F64.s3: List<Double> get() = data
private val S.F8E4M3.s3: List<F8E4M3> get() = data

private val S.U8.dst: MutableList<UByte> get() = data.toMutableList()
private val S.U32.dst: MutableList<UInt> get() = data.toMutableList()
private val S.I16.dst: MutableList<Short> get() = data.toMutableList()
private val S.I32.dst: MutableList<Int> get() = data.toMutableList()
private val S.I64.dst: MutableList<Long> get() = data.toMutableList()
private val S.BF16.dst: MutableList<BF16> get() = data.toMutableList()
private val S.F16.dst: MutableList<F16> get() = data.toMutableList()
private val S.F32.dst: MutableList<Float> get() = data.toMutableList()
private val S.F64.dst: MutableList<Double> get() = data.toMutableList()
private val S.F8E4M3.dst: MutableList<F8E4M3> get() = data.toMutableList()

private val S.U8.src: List<UByte> get() = data
private val S.U32.src: List<UInt> get() = data
private val S.I16.src: List<Short> get() = data
private val S.I32.src: List<Int> get() = data
private val S.I64.src: List<Long> get() = data
private val S.BF16.src: List<BF16> get() = data
private val S.F16.src: List<F16> get() = data
private val S.F32.src: List<Float> get() = data
private val S.F64.src: List<Double> get() = data
private val S.F8E4M3.src: List<F8E4M3> get() = data

public interface Map1 {
    public fun <T> f(
        src: List<T>,
        dev: CudaDevice,
        layout: Layout,
    ): List<T>

    public fun map(s: S, d: CudaDevice, l: Layout, self: Map1 = this): S {
        val out = when (s) {
            is S.U8 -> S.U8(self.f(s.s, d, l))
            is S.U32 -> S.U32(self.f(s.s, d, l))
            is S.I16 -> S.I16(self.f(s.s, d, l))
            is S.I32 -> S.I32(self.f(s.s, d, l))
            is S.I64 -> S.I64(self.f(s.s, d, l))
            is S.BF16 -> S.BF16(self.f(s.s, d, l))
            is S.F16 -> S.F16(self.f(s.s, d, l))
            is S.F32 -> S.F32(self.f(s.s, d, l))
            is S.F64 -> S.F64(self.f(s.s, d, l))
            is S.F8E4M3 -> S.F8E4M3(self.f(s.s, d, l))
            is S.F4, is S.F6E2M3, is S.F6E3M2, is S.F8E8M0 -> {
                bail("Map1 does not uspport this dtype.")
            }
        }
        return Ok(out)
    }
}

public interface Map2 {
    public fun <T> f(
        src1: List<T>,
        layout1: Layout,
        src2: List<T>,
        layout2: Layout,
        dev: CudaDevice,
    ): List<T>

    public fun map(s1: S, l1: Layout, s2: S, l2: Layout, d: CudaDevice, self: Map2 = this): S {
        val out = when {
            s1 is S.U8 && s2 is S.U8 -> S.U8(self.f(s1.s1, l1, s2.s2, l2, d))
            s1 is S.U32 && s2 is S.U32 -> S.U32(self.f(s1.s1, l1, s2.s2, l2, d))
            s1 is S.I16 && s2 is S.I16 -> S.I16(self.f(s1.s1, l1, s2.s2, l2, d))
            s1 is S.I32 && s2 is S.I32 -> S.I32(self.f(s1.s1, l1, s2.s2, l2, d))
            s1 is S.I64 && s2 is S.I64 -> S.I64(self.f(s1.s1, l1, s2.s2, l2, d))
            s1 is S.BF16 && s2 is S.BF16 -> S.BF16(self.f(s1.s1, l1, s2.s2, l2, d))
            s1 is S.F16 && s2 is S.F16 -> S.F16(self.f(s1.s1, l1, s2.s2, l2, d))
            s1 is S.F32 && s2 is S.F32 -> S.F32(self.f(s1.s1, l1, s2.s2, l2, d))
            s1 is S.F64 && s2 is S.F64 -> S.F64(self.f(s1.s1, l1, s2.s2, l2, d))
            s1 is S.F8E4M3 && s2 is S.F8E4M3 -> S.F8E4M3(self.f(s1.s1, l1, s2.s2, l2, d))
            else -> throw Err(CudaError.InternalError("dtype mismatch in binary op"))
        }
        return Ok(out)
    }
}

public interface Map3 {
    public fun <T> f(
        src1: List<T>,
        layout1: Layout,
        src2: List<T>,
        layout2: Layout,
        src3: List<T>,
        layout3: Layout,
        dev: CudaDevice,
    ): List<T>

    public fun map(
        s1: S,
        l1: Layout,
        s2: S,
        l2: Layout,
        s3: S,
        l3: Layout,
        d: CudaDevice,
        self: Map3 = this,
    ): S {
        val out = when {
            s1 is S.U8 && s2 is S.U8 && s3 is S.U8 -> S.U8(self.f(s1.s1, l1, s2.s2, l2, s3.s3, l3, d))
            s1 is S.U32 && s2 is S.U32 && s3 is S.U32 -> S.U32(self.f(s1.s1, l1, s2.s2, l2, s3.s3, l3, d))
            s1 is S.I64 && s2 is S.I64 && s3 is S.I64 -> S.I64(self.f(s1.s1, l1, s2.s2, l2, s3.s3, l3, d))
            s1 is S.BF16 && s2 is S.BF16 && s3 is S.BF16 -> S.BF16(self.f(s1.s1, l1, s2.s2, l2, s3.s3, l3, d))
            s1 is S.F16 && s2 is S.F16 && s3 is S.F16 -> S.F16(self.f(s1.s1, l1, s2.s2, l2, s3.s3, l3, d))
            s1 is S.F32 && s2 is S.F32 && s3 is S.F32 -> S.F32(self.f(s1.s1, l1, s2.s2, l2, s3.s3, l3, d))
            s1 is S.F64 && s2 is S.F64 && s3 is S.F64 -> S.F64(self.f(s1.s1, l1, s2.s2, l2, s3.s3, l3, d))
            s1 is S.F8E4M3 && s2 is S.F8E4M3 && s3 is S.F8E4M3 -> {
                S.F8E4M3(self.f(s1.s1, l1, s2.s2, l2, s3.s3, l3, d))
            }
            else -> throw Err(CudaError.InternalError("dtype mismatch in ternary op"))
        }
        return Ok(out)
    }
}

public interface Map2InPlace {
    public fun <T> f(
        dst: MutableList<T>,
        dstL: Layout,
        src: List<T>,
        srcL: Layout,
        dev: CudaDevice,
    )

    public fun map(
        dst: S,
        dstL: Layout,
        src: S,
        srcL: Layout,
        d: CudaDevice,
        self: Map2InPlace = this,
    ) {
        when {
            dst is S.U8 && src is S.U8 -> self.f(dst.dst, dstL, src.src, srcL, d)
            dst is S.U32 && src is S.U32 -> self.f(dst.dst, dstL, src.src, srcL, d)
            dst is S.I16 && src is S.I16 -> self.f(dst.dst, dstL, src.src, srcL, d)
            dst is S.I32 && src is S.I32 -> self.f(dst.dst, dstL, src.src, srcL, d)
            dst is S.I64 && src is S.I64 -> self.f(dst.dst, dstL, src.src, srcL, d)
            dst is S.BF16 && src is S.BF16 -> self.f(dst.dst, dstL, src.src, srcL, d)
            dst is S.F16 && src is S.F16 -> self.f(dst.dst, dstL, src.src, srcL, d)
            dst is S.F32 && src is S.F32 -> self.f(dst.dst, dstL, src.src, srcL, d)
            dst is S.F64 && src is S.F64 -> self.f(dst.dst, dstL, src.src, srcL, d)
            dst is S.F8E4M3 && src is S.F8E4M3 -> self.f(dst.dst, dstL, src.src, srcL, d)
            else -> throw Err(CudaError.InternalError("dtype mismatch in binary op"))
        }
    }
}

public interface Map1Any {
    public fun <T> f(
        src: List<T>,
        dev: CudaDevice,
        layout: Layout,
        wrap: (List<T>) -> S,
    ): S

    public fun map(s: S, d: CudaDevice, l: Layout, self: Map1Any = this): S {
        val out = when (s) {
            is S.U8 -> self.f(s.s, d, l, S::U8)
            is S.U32 -> self.f(s.s, d, l, S::U32)
            is S.I16 -> self.f(s.s, d, l, S::I16)
            is S.I32 -> self.f(s.s, d, l, S::I32)
            is S.I64 -> self.f(s.s, d, l, S::I64)
            is S.BF16 -> self.f(s.s, d, l, S::BF16)
            is S.F16 -> self.f(s.s, d, l, S::F16)
            is S.F32 -> self.f(s.s, d, l, S::F32)
            is S.F64 -> self.f(s.s, d, l, S::F64)
            is S.F8E4M3 -> self.f(s.s, d, l, S::F8E4M3)
            is S.F4, is S.F6E2M3, is S.F6E3M2, is S.F8E8M0 -> {
                bail("Map1 does not uspport this dtype.")
            }
        }
        return Ok(out)
    }
}

public interface Map2Any {
    public fun <T> f(
        src1: List<T>,
        layout1: Layout,
        src2: List<T>,
        layout2: Layout,
        dev: CudaDevice,
    ): S

    public fun map(s1: S, l1: Layout, s2: S, l2: Layout, d: CudaDevice, self: Map2Any = this): S {
        val out = when {
            s1 is S.U8 && s2 is S.U8 -> self.f(s1.s1, l1, s2.s2, l2, d)
            s1 is S.U32 && s2 is S.U32 -> self.f(s1.s1, l1, s2.s2, l2, d)
            s1 is S.I64 && s2 is S.I64 -> self.f(s1.s1, l1, s2.s2, l2, d)
            s1 is S.BF16 && s2 is S.BF16 -> self.f(s1.s1, l1, s2.s2, l2, d)
            s1 is S.F16 && s2 is S.F16 -> self.f(s1.s1, l1, s2.s2, l2, d)
            s1 is S.F32 && s2 is S.F32 -> self.f(s1.s1, l1, s2.s2, l2, d)
            s1 is S.F64 && s2 is S.F64 -> self.f(s1.s1, l1, s2.s2, l2, d)
            s1 is S.F8E4M3 && s2 is S.F8E4M3 -> self.f(s1.s1, l1, s2.s2, l2, d)
            else -> throw Err(CudaError.InternalError("dtype mismatch in binary op")).w()
        }
        return Ok(out)
    }
}
