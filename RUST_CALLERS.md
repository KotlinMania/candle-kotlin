# Candle canonical package callers

The Rust crate root uses `pub use` to expose symbols from implementation modules.
Kotlin callers import the defining package directly. Do not add `Lib.kt`, `Mod.kt`,
forwarding exports, or compatibility type aliases to reproduce Rust module syntax.
`Module.kt` and `ToUsize2.kt` contain real crate-root implementations and remain.

## Removed aliases

| Former Candle name | Defining Kotlin type |
| --- | --- |
| `Float16`, `F16` | `io.github.kotlinmania.half.F16` |
| `BFloat16`, `BF16` | `io.github.kotlinmania.half.BF16` |
| `F8E4M3`, `F8E5M2` | Corresponding types in `io.github.kotlinmania.float8` |
| `CandleError` | `io.github.kotlinmania.candle.CandleException` |
| `ShapeDim` | `io.github.kotlinmania.candle.Dim` |
| `MatMulUnexpectedStridingDetails` | `io.github.kotlinmania.candle.MatMulUnexpectedStriding` |
| `GgmlTypeCompanion` | `io.github.kotlinmania.candle.GgmlType` |
| `Result` | `kotlin.Result` |
| `Item` | `kotlin.IntArray` |
| `U8Scalar`, `U32Scalar` | `kotlin.UByte`, `kotlin.UInt` |
| `I16Scalar`, `I32Scalar`, `I64Scalar` | `kotlin.Short`, `kotlin.Int`, `kotlin.Long` |
| `F32Scalar`, `F64Scalar` | `kotlin.Float`, `kotlin.Double` |
| `LittleEndian`, `BigEndian` | Corresponding types in `io.github.kotlinmania.byteorder` |
| `CpuStorage.F8E4M3Storage`, `CpuStorage.F8E8M0Storage` | `CpuStorage.F8E4M3`, `CpuStorage.F8E8M0` |
| `CudaStorageSlice.F8E4M3Slice` | `CudaStorageSlice.F8E4M3` |
| `cuda.utils.S` | `io.github.kotlinmania.candle.cuda.CudaStorageSlice` |

`CpuDevice.kt` imports `rng()` from its definition in
`io.github.kotlinmania.rand.rngs`, rather than the Rand crate-root forwarding export.

## Sibling source consumers

The 2026-10-18 workspace source inventory found Candle imports in Candle-NN,
Candle-Transformers, and Moshi. No earlier `RUST_CALLERS.md` was present in Candle,
Candle-NN, or Candle-Transformers.

| Sibling | Source changes required by this migration |
| --- | --- |
| `candle-nn-kotlin` | Replace `ShapeDim` with `Dim` in `candlenn/Ops.kt` and `candlenn/Sampling.kt`; the existing [Candle-NN PR #3](https://github.com/KotlinMania/candle-nn-kotlin/pull/3) carries these edits. |
| `candle-transformers-kotlin` | No references to the removed aliases found in the available source. |
| `moshi-kotlin` | No references to the removed aliases found in the available source. |

These are source-search results, not claims that downstream releases or CI passed.
Before releasing Candle-NN, update its Candle dependency to the validated published
Candle version and run its own build and tests. Do not recreate aliases to avoid
updating callers.
