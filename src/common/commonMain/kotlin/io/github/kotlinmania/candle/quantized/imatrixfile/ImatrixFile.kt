// port-lint: source quantized/imatrix_file.rs
package io.github.kotlinmania.candle.quantized.imatrixfile

import io.github.kotlinmania.byteorder.LittleEndian
import io.github.kotlinmania.candle.error.Error
import io.github.kotlinmania.candle.error.Result
import io.github.kotlinmania.candle.error.bail
import io.github.kotlinmania.rustlang.core.result.fromResidual
import io.github.kotlinmania.rustlang.std.fs.File
import io.github.kotlinmania.rustlang.std.io.Cursor
import io.github.kotlinmania.rustlang.std.io.Read
import io.github.kotlinmania.rustlang.std.path.Path

public fun <P : AsRef<Path>> loadImatrix(fname: P): Result<Map<String, List<Float>>> {
    val allData = HashMap.new()
    var file =
        when (
            val tryResult =
                File.open(fname).mapErr({ e ->
                    Error.msg("Failed to open ${fname.asRef().display()}: $e")
                })
        ) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
    val buffer = mutableListOf()
    when (
        val tryResult =
            file.readToEnd(buffer).mapErr({ e ->
                Error.msg("Failed to read file ${fname.asRef().display()}: $e")
            })
    ) {
        is Result.Ok -> tryResult.value
        is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
    }
    val cursor = Cursor.new(buffer)
    val nEntries =
        (
            when (val tryResult = cursor.readI32<LittleEndian>().mapErr({ e -> Error.msg("Failed to read number of entries: $e") })) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        ).toInt()
    if (nEntries < 1) {
        bail("No data in file {}", fname.asRef().display())
    }
    for (i in 0 until nEntries) {
        // Read length of the name
        val len =
            (
                when (
                    val tryResult =
                        cursor.readI32<LittleEndian>().mapErr({ e ->
                            Error.msg("Failed to read name length for entry ${i + 1}: $e")
                        })
                ) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            ).toInt()
        // Read the name
        val nameBuf = mutableListOf(0u.toUByte(), len)
        when (
            val tryResult =
                cursor.readExact(nameBuf).mapErr({ e ->
                    Error.msg("Failed to read name for entry ${i + 1}: $e")
                })
        ) {
            is Result.Ok -> tryResult.value
            is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
        }
        val name =
            when (
                val tryResult =
                    String.fromUtf8(nameBuf).mapErr({ e ->
                        Error.msg("Invalid UTF-8 name for entry ${i + 1}: $e")
                    })
            ) {
                is Result.Ok -> tryResult.value
                is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
            }
        // Read ncall and nval
        val ncall =
            (
                when (
                    val tryResult =
                        cursor.readI32<LittleEndian>().mapErr({ e ->
                            Error.msg("Failed to read ncall for entry ${i + 1}: $e")
                        })
                ) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            ).toInt()
        val nval =
            (
                when (
                    val tryResult =
                        cursor.readI32<LittleEndian>().mapErr({ e ->
                            Error.msg("Failed to read nval for entry ${i + 1}: $e")
                        })
                ) {
                    is Result.Ok -> tryResult.value
                    is Result.Err -> return Result.fromResidual(Result.Err(tryResult.value))
                }
            ).toInt()
        if (nval < 1) {
            bail("Invalid nval for entry {}: {}", i + 1, nval)
        }
        val data = ArrayList(nval)
        for (unused in 0 until nval) {
            val v = cursor.readF32<LittleEndian>().unwrap()
            if (ncall == 0) {
                data.add(v)
            } else {
                data.add(v / ncall.toFloat())
            }
        }
        allData.insert(name, data)
    }
    return Ok(allData)
}
