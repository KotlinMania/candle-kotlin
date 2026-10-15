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

public interface Map1 {
    public fun <T> f(
        src: List<T>,
        dev: CudaDevice,
        layout: Layout,
    ): List<T>

    public fun map(s: S, d: CudaDevice, l: Layout, self: Map1 = this): S {
        val out = when (s) {
            is S.U8 -> S.U8(self.f(s.data, d, l))
            is S.U32 -> S.U32(self.f(s.data, d, l))
            is S.I16 -> S.I16(self.f(s.data, d, l))
            is S.I32 -> S.I32(self.f(s.data, d, l))
            is S.I64 -> S.I64(self.f(s.data, d, l))
            is S.BF16 -> S.BF16(self.f(s.data, d, l))
            is S.F16 -> S.F16(self.f(s.data, d, l))
            is S.F32 -> S.F32(self.f(s.data, d, l))
            is S.F64 -> S.F64(self.f(s.data, d, l))
            is S.F8E4M3 -> S.F8E4M3(self.f(s.data, d, l))
            is S.F4, is S.F6E2M3, is S.F6E3M2, is S.F8E8M0 -> {
                bail("Map1 does not uspport this dtype.")
            }
        }
        return out
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
            s1 is S.U8 && s2 is S.U8 -> S.U8(self.f(s1.data, l1, s2.data, l2, d))
            s1 is S.U32 && s2 is S.U32 -> S.U32(self.f(s1.data, l1, s2.data, l2, d))
            s1 is S.I16 && s2 is S.I16 -> S.I16(self.f(s1.data, l1, s2.data, l2, d))
            s1 is S.I32 && s2 is S.I32 -> S.I32(self.f(s1.data, l1, s2.data, l2, d))
            s1 is S.I64 && s2 is S.I64 -> S.I64(self.f(s1.data, l1, s2.data, l2, d))
            s1 is S.BF16 && s2 is S.BF16 -> S.BF16(self.f(s1.data, l1, s2.data, l2, d))
            s1 is S.F16 && s2 is S.F16 -> S.F16(self.f(s1.data, l1, s2.data, l2, d))
            s1 is S.F32 && s2 is S.F32 -> S.F32(self.f(s1.data, l1, s2.data, l2, d))
            s1 is S.F64 && s2 is S.F64 -> S.F64(self.f(s1.data, l1, s2.data, l2, d))
            s1 is S.F8E4M3 && s2 is S.F8E4M3 -> S.F8E4M3(self.f(s1.data, l1, s2.data, l2, d))
            else -> throw CudaError.InternalError("dtype mismatch in binary op")
        }
        return out
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
            s1 is S.U8 && s2 is S.U8 && s3 is S.U8 -> S.U8(self.f(s1.data, l1, s2.data, l2, s3.data, l3, d))
            s1 is S.U32 && s2 is S.U32 && s3 is S.U32 -> S.U32(self.f(s1.data, l1, s2.data, l2, s3.data, l3, d))
            s1 is S.I64 && s2 is S.I64 && s3 is S.I64 -> S.I64(self.f(s1.data, l1, s2.data, l2, s3.data, l3, d))
            s1 is S.BF16 && s2 is S.BF16 && s3 is S.BF16 -> S.BF16(self.f(s1.data, l1, s2.data, l2, s3.data, l3, d))
            s1 is S.F16 && s2 is S.F16 && s3 is S.F16 -> S.F16(self.f(s1.data, l1, s2.data, l2, s3.data, l3, d))
            s1 is S.F32 && s2 is S.F32 && s3 is S.F32 -> S.F32(self.f(s1.data, l1, s2.data, l2, s3.data, l3, d))
            s1 is S.F64 && s2 is S.F64 && s3 is S.F64 -> S.F64(self.f(s1.data, l1, s2.data, l2, s3.data, l3, d))
            s1 is S.F8E4M3 && s2 is S.F8E4M3 && s3 is S.F8E4M3 -> {
                S.F8E4M3(self.f(s1.data, l1, s2.data, l2, s3.data, l3, d))
            }
            else -> throw CudaError.InternalError("dtype mismatch in ternary op")
        }
        return out
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
            dst is S.U8 && src is S.U8 -> self.f(dst.data.toMutableList(), dstL, src.data, srcL, d)
            dst is S.U32 && src is S.U32 -> self.f(dst.data.toMutableList(), dstL, src.data, srcL, d)
            dst is S.I16 && src is S.I16 -> self.f(dst.data.toMutableList(), dstL, src.data, srcL, d)
            dst is S.I32 && src is S.I32 -> self.f(dst.data.toMutableList(), dstL, src.data, srcL, d)
            dst is S.I64 && src is S.I64 -> self.f(dst.data.toMutableList(), dstL, src.data, srcL, d)
            dst is S.BF16 && src is S.BF16 -> self.f(dst.data.toMutableList(), dstL, src.data, srcL, d)
            dst is S.F16 && src is S.F16 -> self.f(dst.data.toMutableList(), dstL, src.data, srcL, d)
            dst is S.F32 && src is S.F32 -> self.f(dst.data.toMutableList(), dstL, src.data, srcL, d)
            dst is S.F64 && src is S.F64 -> self.f(dst.data.toMutableList(), dstL, src.data, srcL, d)
            dst is S.F8E4M3 && src is S.F8E4M3 -> self.f(dst.data.toMutableList(), dstL, src.data, srcL, d)
            else -> throw CudaError.InternalError("dtype mismatch in binary op")
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
            is S.U8 -> self.f(s.data, d, l, S::U8)
            is S.U32 -> self.f(s.data, d, l, S::U32)
            is S.I16 -> self.f(s.data, d, l, S::I16)
            is S.I32 -> self.f(s.data, d, l, S::I32)
            is S.I64 -> self.f(s.data, d, l, S::I64)
            is S.BF16 -> self.f(s.data, d, l, S::BF16)
            is S.F16 -> self.f(s.data, d, l, S::F16)
            is S.F32 -> self.f(s.data, d, l, S::F32)
            is S.F64 -> self.f(s.data, d, l, S::F64)
            is S.F8E4M3 -> self.f(s.data, d, l, S::F8E4M3)
            is S.F4, is S.F6E2M3, is S.F6E3M2, is S.F8E8M0 -> {
                bail("Map1 does not uspport this dtype.")
            }
        }
        return out
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
            s1 is S.U8 && s2 is S.U8 -> self.f(s1.data, l1, s2.data, l2, d)
            s1 is S.U32 && s2 is S.U32 -> self.f(s1.data, l1, s2.data, l2, d)
            s1 is S.I64 && s2 is S.I64 -> self.f(s1.data, l1, s2.data, l2, d)
            s1 is S.BF16 && s2 is S.BF16 -> self.f(s1.data, l1, s2.data, l2, d)
            s1 is S.F16 && s2 is S.F16 -> self.f(s1.data, l1, s2.data, l2, d)
            s1 is S.F32 && s2 is S.F32 -> self.f(s1.data, l1, s2.data, l2, d)
            s1 is S.F64 && s2 is S.F64 -> self.f(s1.data, l1, s2.data, l2, d)
            s1 is S.F8E4M3 && s2 is S.F8E4M3 -> self.f(s1.data, l1, s2.data, l2, d)
            else -> throw CudaError.InternalError("dtype mismatch in binary op")
        }
        return out
    }
}
