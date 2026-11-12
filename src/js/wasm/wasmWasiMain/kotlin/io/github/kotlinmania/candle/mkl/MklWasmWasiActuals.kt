@file:OptIn(kotlin.ExperimentalUnsignedTypes::class, kotlin.wasm.ExperimentalWasmInterop::class)

package io.github.kotlinmania.candle.mkl

import io.github.kotlinmania.candle.COpaquePointer
import io.github.kotlinmania.libc.CInt

// The host must provide the C companion wrapper exports in import module candle.
public actual fun vsTanh(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_vsTanh")
}

public actual fun vdTanh(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_vdTanh")
}

public actual fun vsExp(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_vsExp")
}

public actual fun vdExp(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_vdExp")
}

public actual fun vsLn(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_vsLn")
}

public actual fun vdLn(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_vdLn")
}

public actual fun vsSin(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_vsSin")
}

public actual fun vdSin(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_vdSin")
}

public actual fun vsCos(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_vsCos")
}

public actual fun vdCos(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_vdCos")
}

public actual fun vsSqrt(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_vsSqrt")
}

public actual fun vdSqrt(n: CInt, a: COpaquePointer?, y: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_vdSqrt")
}

public actual fun vsAdd(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_vsAdd")
}

public actual fun vdAdd(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_vdAdd")
}

public actual fun vsSub(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_vsSub")
}

public actual fun vdSub(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_vdSub")
}

public actual fun vsMul(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_vsMul")
}

public actual fun vdMul(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_vdMul")
}

public actual fun vsDiv(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_vsDiv")
}

public actual fun vdDiv(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_vdDiv")
}

public actual fun vsFmax(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_vsFmax")
}

public actual fun vdFmax(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_vdFmax")
}

public actual fun vsFmin(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_vsFmin")
}

public actual fun vdFmin(n: CInt, a: COpaquePointer?, b: COpaquePointer?, y: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_vdFmin")
}

public actual fun sgemm(transa: String?, transb: String?, m: COpaquePointer?, n: COpaquePointer?, k: COpaquePointer?, alpha: COpaquePointer?, a: COpaquePointer?, lda: COpaquePointer?, b: COpaquePointer?, ldb: COpaquePointer?, beta: COpaquePointer?, c: COpaquePointer?, ldc: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_sgemm")
}

public actual fun dgemm(transa: String?, transb: String?, m: COpaquePointer?, n: COpaquePointer?, k: COpaquePointer?, alpha: COpaquePointer?, a: COpaquePointer?, lda: COpaquePointer?, b: COpaquePointer?, ldb: COpaquePointer?, beta: COpaquePointer?, c: COpaquePointer?, ldc: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_dgemm")
}

public actual fun hgemm(transa: String?, transb: String?, m: COpaquePointer?, n: COpaquePointer?, k: COpaquePointer?, alpha: COpaquePointer?, a: COpaquePointer?, lda: COpaquePointer?, b: COpaquePointer?, ldb: COpaquePointer?, beta: COpaquePointer?, c: COpaquePointer?, ldc: COpaquePointer?) {
    throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candle_mkl_hgemm")
}
