// port-lint: source cuda_backend/utils.rs
package io.github.kotlinmania.candle.cuda.utils

import io.github.kotlinmania.candle.BF16
import io.github.kotlinmania.candle.F16
import io.github.kotlinmania.candle.F8E4M3
import io.github.kotlinmania.candle.Layout
import io.github.kotlinmania.candle.bail
import io.github.kotlinmania.candle.cuda.CudaStorageSlice
import io.github.kotlinmania.candle.cuda.device.CudaDevice
import io.github.kotlinmania.candle.cuda.device.CudaSlice
import io.github.kotlinmania.candle.cuda.error.CudaError

/** Helper functions to plug cuda kernels in candle. */
public typealias S = CudaStorageSlice

private fun <T> Ok(v: T): T = v
private fun <T : Throwable> Err(v: T): T = v
private fun <T : Throwable> T.w(): T = this

private val CudaStorageSlice.U8.s: CudaSlice<UByte> get() = data
private val CudaStorageSlice.U32.s: CudaSlice<UInt> get() = data
private val CudaStorageSlice.I16.s: CudaSlice<Short> get() = data
private val CudaStorageSlice.I32.s: CudaSlice<Int> get() = data
private val CudaStorageSlice.I64.s: CudaSlice<Long> get() = data
private val CudaStorageSlice.BF16.s: CudaSlice<BF16> get() = data
private val CudaStorageSlice.F16.s: CudaSlice<F16> get() = data
private val CudaStorageSlice.F32.s: CudaSlice<Float> get() = data
private val CudaStorageSlice.F64.s: CudaSlice<Double> get() = data
private val CudaStorageSlice.F8E4M3.s: CudaSlice<F8E4M3> get() = data

private val CudaStorageSlice.U8.s1: CudaSlice<UByte> get() = data
private val CudaStorageSlice.U32.s1: CudaSlice<UInt> get() = data
private val CudaStorageSlice.I16.s1: CudaSlice<Short> get() = data
private val CudaStorageSlice.I32.s1: CudaSlice<Int> get() = data
private val CudaStorageSlice.I64.s1: CudaSlice<Long> get() = data
private val CudaStorageSlice.BF16.s1: CudaSlice<BF16> get() = data
private val CudaStorageSlice.F16.s1: CudaSlice<F16> get() = data
private val CudaStorageSlice.F32.s1: CudaSlice<Float> get() = data
private val CudaStorageSlice.F64.s1: CudaSlice<Double> get() = data
private val CudaStorageSlice.F8E4M3.s1: CudaSlice<F8E4M3> get() = data

private val CudaStorageSlice.U8.s2: CudaSlice<UByte> get() = data
private val CudaStorageSlice.U32.s2: CudaSlice<UInt> get() = data
private val CudaStorageSlice.I16.s2: CudaSlice<Short> get() = data
private val CudaStorageSlice.I32.s2: CudaSlice<Int> get() = data
private val CudaStorageSlice.I64.s2: CudaSlice<Long> get() = data
private val CudaStorageSlice.BF16.s2: CudaSlice<BF16> get() = data
private val CudaStorageSlice.F16.s2: CudaSlice<F16> get() = data
private val CudaStorageSlice.F32.s2: CudaSlice<Float> get() = data
private val CudaStorageSlice.F64.s2: CudaSlice<Double> get() = data
private val CudaStorageSlice.F8E4M3.s2: CudaSlice<F8E4M3> get() = data

private val CudaStorageSlice.U8.s3: CudaSlice<UByte> get() = data
private val CudaStorageSlice.U32.s3: CudaSlice<UInt> get() = data
private val CudaStorageSlice.I64.s3: CudaSlice<Long> get() = data
private val CudaStorageSlice.BF16.s3: CudaSlice<BF16> get() = data
private val CudaStorageSlice.F16.s3: CudaSlice<F16> get() = data
private val CudaStorageSlice.F32.s3: CudaSlice<Float> get() = data
private val CudaStorageSlice.F64.s3: CudaSlice<Double> get() = data
private val CudaStorageSlice.F8E4M3.s3: CudaSlice<F8E4M3> get() = data

private val CudaStorageSlice.U8.dst: CudaSlice<UByte> get() = data
private val CudaStorageSlice.U32.dst: CudaSlice<UInt> get() = data
private val CudaStorageSlice.I16.dst: CudaSlice<Short> get() = data
private val CudaStorageSlice.I32.dst: CudaSlice<Int> get() = data
private val CudaStorageSlice.I64.dst: CudaSlice<Long> get() = data
private val CudaStorageSlice.BF16.dst: CudaSlice<BF16> get() = data
private val CudaStorageSlice.F16.dst: CudaSlice<F16> get() = data
private val CudaStorageSlice.F32.dst: CudaSlice<Float> get() = data
private val CudaStorageSlice.F64.dst: CudaSlice<Double> get() = data
private val CudaStorageSlice.F8E4M3.dst: CudaSlice<F8E4M3> get() = data

private val CudaStorageSlice.U8.src: CudaSlice<UByte> get() = data
private val CudaStorageSlice.U32.src: CudaSlice<UInt> get() = data
private val CudaStorageSlice.I16.src: CudaSlice<Short> get() = data
private val CudaStorageSlice.I32.src: CudaSlice<Int> get() = data
private val CudaStorageSlice.I64.src: CudaSlice<Long> get() = data
private val CudaStorageSlice.BF16.src: CudaSlice<BF16> get() = data
private val CudaStorageSlice.F16.src: CudaSlice<F16> get() = data
private val CudaStorageSlice.F32.src: CudaSlice<Float> get() = data
private val CudaStorageSlice.F64.src: CudaSlice<Double> get() = data
private val CudaStorageSlice.F8E4M3.src: CudaSlice<F8E4M3> get() = data

public interface Map1 {
    public fun <T> f(
        src: CudaSlice<T>,
        dev: CudaDevice,
        layout: Layout,
    ): CudaSlice<T>

    public fun map(s: S, d: CudaDevice, l: Layout, self: Map1 = this): S {
        val out = when (s) {
            is CudaStorageSlice.U8 -> CudaStorageSlice.U8(self.f(s.s, d, l))
            is CudaStorageSlice.U32 -> CudaStorageSlice.U32(self.f(s.s, d, l))
            is CudaStorageSlice.I16 -> CudaStorageSlice.I16(self.f(s.s, d, l))
            is CudaStorageSlice.I32 -> CudaStorageSlice.I32(self.f(s.s, d, l))
            is CudaStorageSlice.I64 -> CudaStorageSlice.I64(self.f(s.s, d, l))
            is CudaStorageSlice.BF16 -> CudaStorageSlice.BF16(self.f(s.s, d, l))
            is CudaStorageSlice.F16 -> CudaStorageSlice.F16(self.f(s.s, d, l))
            is CudaStorageSlice.F32 -> CudaStorageSlice.F32(self.f(s.s, d, l))
            is CudaStorageSlice.F64 -> CudaStorageSlice.F64(self.f(s.s, d, l))
            is CudaStorageSlice.F8E4M3 -> CudaStorageSlice.F8E4M3(self.f(s.s, d, l))
            is CudaStorageSlice.F4, is CudaStorageSlice.F6E2M3, is CudaStorageSlice.F6E3M2, is CudaStorageSlice.F8E8M0 -> {
                bail("Map1 does not uspport this dtype.")
            }
        }
        return Ok(out)
    }
}

public interface Map2 {
    public fun <T> f(
        src1: CudaSlice<T>,
        layout1: Layout,
        src2: CudaSlice<T>,
        layout2: Layout,
        dev: CudaDevice,
    ): CudaSlice<T>

    public fun map(s1: S, l1: Layout, s2: S, l2: Layout, d: CudaDevice, self: Map2 = this): S {
        val out = when {
            s1 is CudaStorageSlice.U8 && s2 is CudaStorageSlice.U8 -> CudaStorageSlice.U8(self.f(s1.s1, l1, s2.s2, l2, d))
            s1 is CudaStorageSlice.U32 && s2 is CudaStorageSlice.U32 -> CudaStorageSlice.U32(self.f(s1.s1, l1, s2.s2, l2, d))
            s1 is CudaStorageSlice.I16 && s2 is CudaStorageSlice.I16 -> CudaStorageSlice.I16(self.f(s1.s1, l1, s2.s2, l2, d))
            s1 is CudaStorageSlice.I32 && s2 is CudaStorageSlice.I32 -> CudaStorageSlice.I32(self.f(s1.s1, l1, s2.s2, l2, d))
            s1 is CudaStorageSlice.I64 && s2 is CudaStorageSlice.I64 -> CudaStorageSlice.I64(self.f(s1.s1, l1, s2.s2, l2, d))
            s1 is CudaStorageSlice.BF16 && s2 is CudaStorageSlice.BF16 -> CudaStorageSlice.BF16(self.f(s1.s1, l1, s2.s2, l2, d))
            s1 is CudaStorageSlice.F16 && s2 is CudaStorageSlice.F16 -> CudaStorageSlice.F16(self.f(s1.s1, l1, s2.s2, l2, d))
            s1 is CudaStorageSlice.F32 && s2 is CudaStorageSlice.F32 -> CudaStorageSlice.F32(self.f(s1.s1, l1, s2.s2, l2, d))
            s1 is CudaStorageSlice.F64 && s2 is CudaStorageSlice.F64 -> CudaStorageSlice.F64(self.f(s1.s1, l1, s2.s2, l2, d))
            s1 is CudaStorageSlice.F8E4M3 && s2 is CudaStorageSlice.F8E4M3 -> CudaStorageSlice.F8E4M3(self.f(s1.s1, l1, s2.s2, l2, d))
            else -> throw Err(CudaError.InternalError("dtype mismatch in binary op"))
        }
        return Ok(out)
    }
}

public interface Map3 {
    public fun <T> f(
        src1: CudaSlice<T>,
        layout1: Layout,
        src2: CudaSlice<T>,
        layout2: Layout,
        src3: CudaSlice<T>,
        layout3: Layout,
        dev: CudaDevice,
    ): CudaSlice<T>

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
            s1 is CudaStorageSlice.U8 && s2 is CudaStorageSlice.U8 && s3 is CudaStorageSlice.U8 -> CudaStorageSlice.U8(self.f(s1.s1, l1, s2.s2, l2, s3.s3, l3, d))
            s1 is CudaStorageSlice.U32 && s2 is CudaStorageSlice.U32 && s3 is CudaStorageSlice.U32 -> CudaStorageSlice.U32(self.f(s1.s1, l1, s2.s2, l2, s3.s3, l3, d))
            s1 is CudaStorageSlice.I64 && s2 is CudaStorageSlice.I64 && s3 is CudaStorageSlice.I64 -> CudaStorageSlice.I64(self.f(s1.s1, l1, s2.s2, l2, s3.s3, l3, d))
            s1 is CudaStorageSlice.BF16 && s2 is CudaStorageSlice.BF16 && s3 is CudaStorageSlice.BF16 -> CudaStorageSlice.BF16(self.f(s1.s1, l1, s2.s2, l2, s3.s3, l3, d))
            s1 is CudaStorageSlice.F16 && s2 is CudaStorageSlice.F16 && s3 is CudaStorageSlice.F16 -> CudaStorageSlice.F16(self.f(s1.s1, l1, s2.s2, l2, s3.s3, l3, d))
            s1 is CudaStorageSlice.F32 && s2 is CudaStorageSlice.F32 && s3 is CudaStorageSlice.F32 -> CudaStorageSlice.F32(self.f(s1.s1, l1, s2.s2, l2, s3.s3, l3, d))
            s1 is CudaStorageSlice.F64 && s2 is CudaStorageSlice.F64 && s3 is CudaStorageSlice.F64 -> CudaStorageSlice.F64(self.f(s1.s1, l1, s2.s2, l2, s3.s3, l3, d))
            s1 is CudaStorageSlice.F8E4M3 && s2 is CudaStorageSlice.F8E4M3 && s3 is CudaStorageSlice.F8E4M3 -> {
                CudaStorageSlice.F8E4M3(self.f(s1.s1, l1, s2.s2, l2, s3.s3, l3, d))
            }
            else -> throw Err(CudaError.InternalError("dtype mismatch in ternary op"))
        }
        return Ok(out)
    }
}

public interface Map2InPlace {
    public fun <T> f(
        dst: CudaSlice<T>,
        dstL: Layout,
        src: CudaSlice<T>,
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
            dst is CudaStorageSlice.U8 && src is CudaStorageSlice.U8 -> self.f(dst.dst, dstL, src.src, srcL, d)
            dst is CudaStorageSlice.U32 && src is CudaStorageSlice.U32 -> self.f(dst.dst, dstL, src.src, srcL, d)
            dst is CudaStorageSlice.I16 && src is CudaStorageSlice.I16 -> self.f(dst.dst, dstL, src.src, srcL, d)
            dst is CudaStorageSlice.I32 && src is CudaStorageSlice.I32 -> self.f(dst.dst, dstL, src.src, srcL, d)
            dst is CudaStorageSlice.I64 && src is CudaStorageSlice.I64 -> self.f(dst.dst, dstL, src.src, srcL, d)
            dst is CudaStorageSlice.BF16 && src is CudaStorageSlice.BF16 -> self.f(dst.dst, dstL, src.src, srcL, d)
            dst is CudaStorageSlice.F16 && src is CudaStorageSlice.F16 -> self.f(dst.dst, dstL, src.src, srcL, d)
            dst is CudaStorageSlice.F32 && src is CudaStorageSlice.F32 -> self.f(dst.dst, dstL, src.src, srcL, d)
            dst is CudaStorageSlice.F64 && src is CudaStorageSlice.F64 -> self.f(dst.dst, dstL, src.src, srcL, d)
            dst is CudaStorageSlice.F8E4M3 && src is CudaStorageSlice.F8E4M3 -> self.f(dst.dst, dstL, src.src, srcL, d)
            else -> throw Err(CudaError.InternalError("dtype mismatch in binary op"))
        }
    }
}

public interface Map1Any {
    public fun <T> f(
        src: CudaSlice<T>,
        dev: CudaDevice,
        layout: Layout,
        wrap: (CudaSlice<T>) -> S,
    ): S

    public fun map(s: S, d: CudaDevice, l: Layout, self: Map1Any = this): S {
        val out = when (s) {
            is CudaStorageSlice.U8 -> self.f(s.s, d, l, CudaStorageSlice::U8)
            is CudaStorageSlice.U32 -> self.f(s.s, d, l, CudaStorageSlice::U32)
            is CudaStorageSlice.I16 -> self.f(s.s, d, l, CudaStorageSlice::I16)
            is CudaStorageSlice.I32 -> self.f(s.s, d, l, CudaStorageSlice::I32)
            is CudaStorageSlice.I64 -> self.f(s.s, d, l, CudaStorageSlice::I64)
            is CudaStorageSlice.BF16 -> self.f(s.s, d, l, CudaStorageSlice::BF16)
            is CudaStorageSlice.F16 -> self.f(s.s, d, l, CudaStorageSlice::F16)
            is CudaStorageSlice.F32 -> self.f(s.s, d, l, CudaStorageSlice::F32)
            is CudaStorageSlice.F64 -> self.f(s.s, d, l, CudaStorageSlice::F64)
            is CudaStorageSlice.F8E4M3 -> self.f(s.s, d, l, CudaStorageSlice::F8E4M3)
            is CudaStorageSlice.F4, is CudaStorageSlice.F6E2M3, is CudaStorageSlice.F6E3M2, is CudaStorageSlice.F8E8M0 -> {
                bail("Map1 does not uspport this dtype.")
            }
        }
        return Ok(out)
    }
}

public interface Map2Any {
    public fun <T> f(
        src1: CudaSlice<T>,
        layout1: Layout,
        src2: CudaSlice<T>,
        layout2: Layout,
        dev: CudaDevice,
    ): S

    public fun map(s1: S, l1: Layout, s2: S, l2: Layout, d: CudaDevice, self: Map2Any = this): S {
        val out = when {
            s1 is CudaStorageSlice.U8 && s2 is CudaStorageSlice.U8 -> self.f(s1.s1, l1, s2.s2, l2, d)
            s1 is CudaStorageSlice.U32 && s2 is CudaStorageSlice.U32 -> self.f(s1.s1, l1, s2.s2, l2, d)
            s1 is CudaStorageSlice.I64 && s2 is CudaStorageSlice.I64 -> self.f(s1.s1, l1, s2.s2, l2, d)
            s1 is CudaStorageSlice.BF16 && s2 is CudaStorageSlice.BF16 -> self.f(s1.s1, l1, s2.s2, l2, d)
            s1 is CudaStorageSlice.F16 && s2 is CudaStorageSlice.F16 -> self.f(s1.s1, l1, s2.s2, l2, d)
            s1 is CudaStorageSlice.F32 && s2 is CudaStorageSlice.F32 -> self.f(s1.s1, l1, s2.s2, l2, d)
            s1 is CudaStorageSlice.F64 && s2 is CudaStorageSlice.F64 -> self.f(s1.s1, l1, s2.s2, l2, d)
            s1 is CudaStorageSlice.F8E4M3 && s2 is CudaStorageSlice.F8E4M3 -> self.f(s1.s1, l1, s2.s2, l2, d)
            else -> throw Err(CudaError.InternalError("dtype mismatch in binary op")).w()
        }
        return Ok(out)
    }
}
