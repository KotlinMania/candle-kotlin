@file:OptIn(kotlin.ExperimentalUnsignedTypes::class, kotlin.wasm.ExperimentalWasmInterop::class)

package io.github.kotlinmania.candle.accelerate

import io.github.kotlinmania.candle.COpaquePointer
import io.github.kotlinmania.libc.CLong
import io.github.kotlinmania.libc.CUlong

// The host must provide the C companion wrapper exports in import module candle.
public actual fun sgemmFfi(transa: String?, transb: String?, m: COpaquePointer?, n: COpaquePointer?, k: COpaquePointer?, alpha: COpaquePointer?, a: COpaquePointer?, lda: COpaquePointer?, b: COpaquePointer?, ldb: COpaquePointer?, beta: COpaquePointer?, c: COpaquePointer?, ldc: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_accelerate_sgemmFfi")
}

public actual fun dgemmFfi(transa: String?, transb: String?, m: COpaquePointer?, n: COpaquePointer?, k: COpaquePointer?, alpha: COpaquePointer?, a: COpaquePointer?, lda: COpaquePointer?, b: COpaquePointer?, ldb: COpaquePointer?, beta: COpaquePointer?, c: COpaquePointer?, ldc: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_accelerate_dgemmFfi")
}

public actual fun vvexpf(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_accelerate_vvexpf")
}

public actual fun vvexp(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_accelerate_vvexp")
}

public actual fun vvsqrtf(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_accelerate_vvsqrtf")
}

public actual fun vvsqrt(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_accelerate_vvsqrt")
}

public actual fun vvsinf(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_accelerate_vvsinf")
}

public actual fun vvsin(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_accelerate_vvsin")
}

public actual fun vvcosf(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_accelerate_vvcosf")
}

public actual fun vvcos(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_accelerate_vvcos")
}

public actual fun vvlogf(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_accelerate_vvlogf")
}

public actual fun vvlog(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_accelerate_vvlog")
}

public actual fun vvtanhf(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_accelerate_vvtanhf")
}

public actual fun vvtanh(dst: COpaquePointer?, src: COpaquePointer?, len: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_accelerate_vvtanh")
}

public actual fun vDspVaddD(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_accelerate_vDspVaddD")
}

public actual fun vDspVadd(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_accelerate_vDspVadd")
}

public actual fun vDspVsubD(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_accelerate_vDspVsubD")
}

public actual fun vDspVsub(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_accelerate_vDspVsub")
}

public actual fun vDspVmulD(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_accelerate_vDspVmulD")
}

public actual fun vDspVmul(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_accelerate_vDspVmul")
}

public actual fun vDspVdivD(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_accelerate_vDspVdivD")
}

public actual fun vDspVdiv(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_accelerate_vDspVdiv")
}

public actual fun vDspVminD(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_accelerate_vDspVminD")
}

public actual fun vDspVmin(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_accelerate_vDspVmin")
}

public actual fun vDspVmaxD(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_accelerate_vDspVmaxD")
}

public actual fun vDspVmax(arg0: COpaquePointer?, arg1: CLong, arg2: COpaquePointer?, arg3: CLong, arg4: COpaquePointer?, arg5: CLong, arg6: CUlong) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_accelerate_vDspVmax")
}
