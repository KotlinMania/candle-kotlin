@file:OptIn(kotlin.ExperimentalUnsignedTypes::class, kotlin.wasm.ExperimentalWasmInterop::class)

package io.github.kotlinmania.candle.mkl

import io.github.kotlinmania.libc.CInt
import io.github.kotlinmania.rustlang.core.ptr.ConstPtr
import io.github.kotlinmania.rustlang.core.ptr.MutPtr

// The host must provide the C companion wrapper exports in import module candle.
public actual fun vsTanh(n: CInt, a: ConstPtr, y: MutPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklVsTanh")

public actual fun vdTanh(n: CInt, a: ConstPtr, y: MutPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklVdTanh")

public actual fun vsExp(n: CInt, a: ConstPtr, y: MutPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklVsExp")

public actual fun vdExp(n: CInt, a: ConstPtr, y: MutPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklVdExp")

public actual fun vsLn(n: CInt, a: ConstPtr, y: MutPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklVsLn")

public actual fun vdLn(n: CInt, a: ConstPtr, y: MutPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklVdLn")

public actual fun vsSin(n: CInt, a: ConstPtr, y: MutPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklVsSin")

public actual fun vdSin(n: CInt, a: ConstPtr, y: MutPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklVdSin")

public actual fun vsCos(n: CInt, a: ConstPtr, y: MutPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklVsCos")

public actual fun vdCos(n: CInt, a: ConstPtr, y: MutPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklVdCos")

public actual fun vsSqrt(n: CInt, a: ConstPtr, y: MutPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklVsSqrt")

public actual fun vdSqrt(n: CInt, a: ConstPtr, y: MutPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklVdSqrt")

public actual fun vsAdd(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklVsAdd")

public actual fun vdAdd(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklVdAdd")

public actual fun vsSub(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklVsSub")

public actual fun vdSub(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklVdSub")

public actual fun vsMul(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklVsMul")

public actual fun vdMul(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklVdMul")

public actual fun vsDiv(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklVsDiv")

public actual fun vdDiv(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklVdDiv")

public actual fun vsFmax(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklVsFmax")

public actual fun vdFmax(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklVdFmax")

public actual fun vsFmin(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklVsFmin")

public actual fun vdFmin(n: CInt, a: ConstPtr, b: ConstPtr, y: MutPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklVdFmin")

public actual fun sgemm(transa: ConstPtr, transb: ConstPtr, m: ConstPtr, n: ConstPtr, k: ConstPtr, alpha: ConstPtr, a: ConstPtr, lda: ConstPtr, b: ConstPtr, ldb: ConstPtr, beta: ConstPtr, c: MutPtr, ldc: ConstPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklSgemm")

public actual fun dgemm(transa: ConstPtr, transb: ConstPtr, m: ConstPtr, n: ConstPtr, k: ConstPtr, alpha: ConstPtr, a: ConstPtr, lda: ConstPtr, b: ConstPtr, ldb: ConstPtr, beta: ConstPtr, c: MutPtr, ldc: ConstPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklDgemm")

public actual fun hgemm(transa: ConstPtr, transb: ConstPtr, m: ConstPtr, n: ConstPtr, k: ConstPtr, alpha: ConstPtr, a: ConstPtr, lda: ConstPtr, b: ConstPtr, ldb: ConstPtr, beta: ConstPtr, c: MutPtr, ldc: ConstPtr): Unit = throw UnsupportedOperationException("WasmWasi FFI requires a host memory/callback/aggregate or variadic ABI adapter for candleMklHgemm")
