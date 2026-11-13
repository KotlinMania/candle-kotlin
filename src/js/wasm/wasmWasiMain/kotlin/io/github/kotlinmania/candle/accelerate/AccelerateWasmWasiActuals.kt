@file:OptIn(kotlin.ExperimentalUnsignedTypes::class, kotlin.wasm.ExperimentalWasmInterop::class)

package io.github.kotlinmania.candle.accelerate

import io.github.kotlinmania.libc.CLong
import io.github.kotlinmania.libc.CUlong
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr

// The host must provide the C companion wrapper exports in import module candle.
public actual fun sgemmFfi(transa: ConstPtr, transb: ConstPtr, m: ConstPtr, n: ConstPtr, k: ConstPtr, alpha: ConstPtr, a: ConstPtr, lda: ConstPtr, b: ConstPtr, ldb: ConstPtr, beta: ConstPtr, c: MutPtr, ldc: ConstPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleAccelerateSgemmFfi")

public actual fun dgemmFfi(transa: ConstPtr, transb: ConstPtr, m: ConstPtr, n: ConstPtr, k: ConstPtr, alpha: ConstPtr, a: ConstPtr, lda: ConstPtr, b: ConstPtr, ldb: ConstPtr, beta: ConstPtr, c: MutPtr, ldc: ConstPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleAccelerateDgemmFfi")

public actual fun vvexpf(dst: MutPtr, src: ConstPtr, len: ConstPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleAccelerateVvexpf")

public actual fun vvexp(dst: MutPtr, src: ConstPtr, len: ConstPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleAccelerateVvexp")

public actual fun vvsqrtf(dst: MutPtr, src: ConstPtr, len: ConstPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleAccelerateVvsqrtf")

public actual fun vvsqrt(dst: MutPtr, src: ConstPtr, len: ConstPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleAccelerateVvsqrt")

public actual fun vvsinf(dst: MutPtr, src: ConstPtr, len: ConstPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleAccelerateVvsinf")

public actual fun vvsin(dst: MutPtr, src: ConstPtr, len: ConstPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleAccelerateVvsin")

public actual fun vvcosf(dst: MutPtr, src: ConstPtr, len: ConstPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleAccelerateVvcosf")

public actual fun vvcos(dst: MutPtr, src: ConstPtr, len: ConstPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleAccelerateVvcos")

public actual fun vvlogf(dst: MutPtr, src: ConstPtr, len: ConstPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleAccelerateVvlogf")

public actual fun vvlog(dst: MutPtr, src: ConstPtr, len: ConstPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleAccelerateVvlog")

public actual fun vvtanhf(dst: MutPtr, src: ConstPtr, len: ConstPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleAccelerateVvtanhf")

public actual fun vvtanh(dst: MutPtr, src: ConstPtr, len: ConstPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleAccelerateVvtanh")

public actual fun vDspVaddD(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleAccelerateVDspVaddD")

public actual fun vDspVadd(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleAccelerateVDspVadd")

public actual fun vDspVsubD(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleAccelerateVDspVsubD")

public actual fun vDspVsub(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleAccelerateVDspVsub")

public actual fun vDspVmulD(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleAccelerateVDspVmulD")

public actual fun vDspVmul(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleAccelerateVDspVmul")

public actual fun vDspVdivD(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleAccelerateVDspVdivD")

public actual fun vDspVdiv(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleAccelerateVDspVdiv")

public actual fun vDspVminD(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleAccelerateVDspVminD")

public actual fun vDspVmin(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleAccelerateVDspVmin")

public actual fun vDspVmaxD(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleAccelerateVDspVmaxD")

public actual fun vDspVmax(arg0: ConstPtr, arg1: CLong, arg2: ConstPtr, arg3: CLong, arg4: MutPtr, arg5: CLong, arg6: CUlong): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleAccelerateVDspVmax")
