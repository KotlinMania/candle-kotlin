package io.github.kotlinmania.candle.bilineartests

import io.github.kotlinmania.candlecore.DType
import io.github.kotlinmania.candlecore.Device
import io.github.kotlinmania.candlecore.Result
import io.github.kotlinmania.candlecore.Tensor

// ============================================================================
// PyTorch Exact Comparison Tests
// ============================================================================
// These tests compare against exact PyTorch outputs to ensure correctness
// Test corresponds to PyTorch:
// import torch
// import torch.nn.functional as F
// input = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
// output = F.interpolate(input, size=(8, 8), mode='bilinear', alignCorners=False)
//
public fun bilinearPytorch2xUpscale(dev: Device): Result<Unit> {
    val input =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0.0f, 16.0f, dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(listOf(1, 1, 4, 4))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val output =
        when (val tryResult = input.upsampleBilinear2d(8, 8, false)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // PyTorch expected output (verified from PyTorch 2.10.0)
    val expected =
        when (
            val tryResult =
                when (val tryResult = Tensor.new(listOf(0.0000f, 0.2500, 0.7500, 1.2500, 1.7500, 2.2500, 2.7500, 3.0000, 1.0000, 1.2500, 1.7500, 2.2500, 2.7500, 3.2500, 3.7500, 4.0000, 3.0000, 3.2500, 3.7500, 4.2500, 4.7500, 5.2500, 5.7500, 6.0000, 5.0000, 5.2500, 5.7500, 6.2500, 6.7500, 7.2500, 7.7500, 8.0000, 7.0000, 7.2500, 7.7500, 8.2500, 8.7500, 9.2500, 9.7500, 10.0000, 9.0000, 9.2500, 9.7500, 10.2500, 10.7500, 11.2500, 11.7500, 12.0000, 11.0000, 11.2500, 11.7500, 12.2500, 12.7500, 13.2500, 13.7500, 14.0000, 12.0000, 12.2500, 12.7500, 13.2500, 13.7500, 14.2500, 14.7500, 15.0000), dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(listOf(1, 1, 8, 8))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = (output - expected)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.flattenAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.max(0)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val maxDiff =
        when (val tryResult = diff.toVec0<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(maxDiff < 1e-4) { "Max difference $maxDiff exceeds threshold 1e-4" }
    return Result.Ok(Unit)
}

// Test corresponds to PyTorch:
// import torch
// import torch.nn.functional as F
// input = torch.arange(64, dtype=torch.float32).reshape(1, 1, 8, 8)
// output = F.interpolate(input, size=(4, 4), mode='bilinear', alignCorners=False)
//
public fun bilinearPytorchDownscale(dev: Device): Result<Unit> {
    val input =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0.0f, 64.0f, dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(listOf(1, 1, 8, 8))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val output =
        when (val tryResult = input.upsampleBilinear2d(4, 4, false)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // PyTorch expected output
    val expected =
        when (
            val tryResult =
                when (val tryResult = Tensor.new(listOf(4.5f, 6.5, 8.5, 10.5, 20.5, 22.5, 24.5, 26.5, 36.5, 38.5, 40.5, 42.5, 52.5, 54.5, 56.5, 58.5), dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(listOf(1, 1, 4, 4))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = (output - expected)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.flattenAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.max(0)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val maxDiff =
        when (val tryResult = diff.toVec0<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(maxDiff < 1e-4) { "Max difference $maxDiff exceeds threshold 1e-4" }
    return Result.Ok(Unit)
}

// Test corresponds to PyTorch:
// import torch
// import torch.nn.functional as F
// torch.manualSeed(42)
// input = torch.randn(1, 2, 4, 4, dtype=torch.float32)
// output = F.interpolate(input, size=(8, 8), mode='bilinear', alignCorners=False)
//
public fun bilinearPytorchMultiChannel(dev: Device): Result<Unit> {
    // Using fixed seed data from PyTorch (seed=42)
    val input =
        when (
            val tryResult =
                when (val tryResult = Tensor.new(listOf(1.9269f, 1.4873, 0.9007, -2.1055, 0.6784, -1.2345, -0.0431, -1.6047, -0.7521, 1.6487, -0.3925, -1.4036, -0.7279, -0.5594, -0.7688, 0.7624, 1.6423f, -0.1596, -0.4974, 0.4396, -0.7581, 1.0783, 0.8008, 1.6806, 1.2791, 1.2964, 0.6105, 1.3347, -0.2316, 0.0418, -0.2516, 0.8599), dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(listOf(1, 2, 4, 4))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val output =
        when (val tryResult = input.upsampleBilinear2d(8, 8, false)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(output.dims() == listOf(1, 2, 8, 8))
    // Verify output is finite and in reasonable range
    val outputVec =
        when (
            val tryResult =
                when (val tryResult = output.flattenAll()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    for (value in outputVec) {
        check(value.isFinite()) { "Output contains non-finite value" }
    }
    // Check first row of channel 0 from PyTorch output
    val outputCh0Row0 =
        when (
            val tryResult =
                when (val tryResult = output.i(listOf(0, 0, 0, 0 until 0))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val expectedCh0Row0 = listOf(1.9269f, 1.8170, 1.5972, 1.3406, 1.0474, 0.1492, -1.3540, -2.1055)
    for ((i, out, exp) in (outputCh0Row0.iter().zip(expectedCh0Row0.iter())).mapIndexed { index, item -> Pair(index, item) }) {
        val diff = (out - exp).abs()
        check(diff < 1e-3) { "Channel 0, row 0, index $i differs: got $out, expected $exp, diff $diff" }
    }
    // Check first row of channel 1 from PyTorch output
    val outputCh1Row0 =
        when (
            val tryResult =
                when (val tryResult = output.i(listOf(0, 1, 0, 0 until 0))) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val expectedCh1Row0 = listOf(1.6423f, 1.1918, 0.2909, -0.2440, -0.4129, -0.2632, 0.2053, 0.4396)
    for ((i1, out1, exp1) in (outputCh1Row0.iter().zip(expectedCh1Row0.iter())).mapIndexed { index, item -> Pair(index, item) }) {
        val diff1 = (out1 - exp1).abs()
        check(diff1 < 1e-3) { "Channel 1, row 0, index $i1 differs: got $out1, expected $exp1, diff $diff1" }
    }
    return Result.Ok(Unit)
}

// Test corresponds to PyTorch:
// import torch
// import torch.nn.functional as F
// input = torch.tensor([[[[1.0, 2.0], [3.0, 4.0]]]], dtype=torch.float32)
// output = F.interpolate(input, size=(4, 4), mode='bilinear', alignCorners=True)
//
public fun bilinearPytorchAlignCornersTrue(dev: Device): Result<Unit> {
    val input =
        when (val tryResult = Tensor.fromVec(mutableListOf(1.0f, 2.0, 3.0, 4.0), listOf(1, 1, 2, 2), dev)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val output =
        when (val tryResult = input.upsampleBilinear2d(4, 4, true)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // PyTorch expected output with alignCorners=True
    val expected =
        when (
            val tryResult =
                when (val tryResult = Tensor.new(listOf(1.0f, 1.3333, 1.6667, 2.0, 1.6667, 2.0, 2.3333, 2.6667, 2.3333, 2.6667, 3.0, 3.3333, 3.0, 3.3333, 3.6667, 4.0), dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(listOf(1, 1, 4, 4))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = (output - expected)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.flattenAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.max(0)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val maxDiff =
        when (val tryResult = diff.toVec0<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(maxDiff < 1e-3) { "Max difference $maxDiff exceeds threshold 1e-3" }
    // Verify corners are exactly preserved with alignCorners=True
    val outputVec =
        when (
            val tryResult =
                when (val tryResult = output.flattenAll()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check((outputVec[0] - 1.0).abs() < 1e-5) { "Top-left corner not preserved" }
    check((outputVec[3] - 2.0).abs() < 1e-5) { "Top-right corner not preserved" }
    check((outputVec[12] - 3.0).abs() < 1e-5) { "Bottom-left corner not preserved" }
    check((outputVec[15] - 4.0).abs() < 1e-5) { "Bottom-right corner not preserved" }
    return Result.Ok(Unit)
}

// Test corresponds to PyTorch:
// import torch
// import torch.nn.functional as F
// input = torch.arange(16, dtype=torch.float32).reshape(1, 1, 4, 4)
// output = F.interpolate(input, scaleFactor=2.0, mode='bilinear', alignCorners=False)
//
public fun bilinearPytorchScaleFactor(dev: Device): Result<Unit> {
    val input =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0.0f, 16.0f, dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(listOf(1, 1, 4, 4))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val outputScale =
        when (val tryResult = input.upsampleBilinear2dWithScale(2.0, 2.0, false)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val outputSize =
        when (val tryResult = input.upsampleBilinear2d(8, 8, false)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // scaleFactor=2.0 should produce identical results to size=(8, 8)
    val diff =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = (outputScale - outputSize)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.flattenAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.max(0)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val maxDiff =
        when (val tryResult = diff.toVec0<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(maxDiff < 1e-6) { "scale_factor and size methods differ by $maxDiff" }
    return Result.Ok(Unit)
}

// Test corresponds to PyTorch:
// import torch
// import torch.nn.functional as F
// input = torch.arange(24, dtype=torch.float32).reshape(1, 1, 4, 6)
// output = F.interpolate(input, size=(8, 12), mode='bilinear', alignCorners=False)
//
public fun bilinearPytorchNonSquareExact(dev: Device): Result<Unit> {
    val input =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0.0f, 24.0f, dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(listOf(1, 1, 4, 6))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val output =
        when (val tryResult = input.upsampleBilinear2d(8, 12, false)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // PyTorch expected output (verified from PyTorch 2.10.0)
    val expected =
        when (
            val tryResult =
                when (val tryResult = Tensor.new(listOf(0.0f, 0.25, 0.75, 1.25, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.0, 1.5, 1.75, 2.25, 2.75, 3.25, 3.75, 4.25, 4.75, 5.25, 5.75, 6.25, 6.5, 4.5, 4.75, 5.25, 5.75, 6.25, 6.75, 7.25, 7.75, 8.25, 8.75, 9.25, 9.5, 7.5, 7.75, 8.25, 8.75, 9.25, 9.75, 10.25, 10.75, 11.25, 11.75, 12.25, 12.5, 10.5, 10.75, 11.25, 11.75, 12.25, 12.75, 13.25, 13.75, 14.25, 14.75, 15.25, 15.5, 13.5, 13.75, 14.25, 14.75, 15.25, 15.75, 16.25, 16.75, 17.25, 17.75, 18.25, 18.5, 16.5, 16.75, 17.25, 17.75, 18.25, 18.75, 19.25, 19.75, 20.25, 20.75, 21.25, 21.5, 18.0, 18.25, 18.75, 19.25, 19.75, 20.25, 20.75, 21.25, 21.75, 22.25, 22.75, 23.0), dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(listOf(1, 1, 8, 12))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = (output - expected)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.flattenAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.max(0)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val maxDiff =
        when (val tryResult = diff.toVec0<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(maxDiff < 1e-4) { "Max difference $maxDiff exceeds threshold 1e-4" }
    return Result.Ok(Unit)
}

// Test corresponds to PyTorch:
// import torch
// import torch.nn.functional as F
// input = torch.tensor([[[[5.0]]]], dtype=torch.float32)
// output = F.interpolate(input, size=(3, 3), mode='bilinear', alignCorners=False)
//
public fun bilinearPytorchTiny1x1To3x3(dev: Device): Result<Unit> {
    val input =
        when (
            val tryResult =
                when (val tryResult = Tensor.new(listOf(5.0f), dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(listOf(1, 1, 1, 1))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val output =
        when (val tryResult = input.upsampleBilinear2d(3, 3, false)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // PyTorch expected output: all values should be 5.0
    val expected =
        when (
            val tryResult =
                when (val tryResult = Tensor.new(listOf(5.0f, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0), dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(listOf(1, 1, 3, 3))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = (output - expected)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.flattenAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.max(0)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val maxDiff =
        when (val tryResult = diff.toVec0<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(maxDiff < 1e-6) { "Max difference $maxDiff exceeds threshold 1e-6" }
    return Result.Ok(Unit)
}

// Test corresponds to PyTorch:
// import torch
// import torch.nn.functional as F
// input = torch.tensor([[[[2.0, 8.0]]]], dtype=torch.float32)
// output = F.interpolate(input, size=(3, 6), mode='bilinear', alignCorners=False)
//
public fun bilinearPytorchTiny1x2To3x6(dev: Device): Result<Unit> {
    val input =
        when (
            val tryResult =
                when (val tryResult = Tensor.new(listOf(2.0f, 8.0), dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(listOf(1, 1, 1, 2))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val output =
        when (val tryResult = input.upsampleBilinear2d(3, 6, false)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // PyTorch expected output
    val expected =
        when (
            val tryResult =
                when (val tryResult = Tensor.new(listOf(2.0f, 2.0, 4.0, 6.0, 8.0, 8.0, 2.0, 2.0, 4.0, 6.0, 8.0, 8.0, 2.0, 2.0, 4.0, 6.0, 8.0, 8.0), dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(listOf(1, 1, 3, 6))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = (output - expected)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.flattenAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.max(0)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val maxDiff =
        when (val tryResult = diff.toVec0<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(maxDiff < 1e-6) { "Max difference $maxDiff exceeds threshold 1e-6" }
    return Result.Ok(Unit)
}

// Test corresponds to PyTorch:
// import torch
// import torch.nn.functional as F
// torch.manualSeed(123)
// input = torch.randn(1, 1, 64, 64, dtype=torch.float32)
// output = F.interpolate(input, size=(128, 128), mode='bilinear', alignCorners=False)
//
public fun bilinearPytorchLarge64x64To128x128(dev: Device): Result<Unit> {
    val input =
        when (val tryResult = Tensor.randn(0.0f, 1.0f, listOf(1, 1, 64, 64), dev)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val output =
        when (val tryResult = input.upsampleBilinear2d(128, 128, false)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(output.dims() == listOf(1, 1, 128, 128))
    check(output.dtype() == DType.F32)
    // Verify all values are finite
    val outputVec =
        when (
            val tryResult =
                when (val tryResult = output.flattenAll()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    for (value in outputVec) {
        check(value.isFinite()) { "Large tensor output contains non-finite value" }
    }
    // Verify output is in reasonable range (should be similar to input range)
    val minVal = outputVec.iter().copied().fold(Float.INFINITY, Float::min)
    val maxVal = outputVec.iter().copied().fold(Float.NEG_INFINITY, Float::max)
    check(minVal > -10.0 && maxVal < 10.0) { "Large tensor output values out of expected range: min=$minVal, max=$maxVal" }
    return Result.Ok(Unit)
}

// ============================================================================
// Dimension and Shape Tests (Consolidated)
// ============================================================================
// These tests verify correct output dimensions for various input configurations
public fun bilinearOutputDimensions(dev: Device): Result<Unit> {
    // Test 1: Non-square dimensions
    val t1 =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0.0f, 32.0f, dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(listOf(1, 1, 4, 8))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val out1 =
        when (val tryResult = t1.upsampleBilinear2d(6, 12, false)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(out1.dims() == listOf(1, 1, 6, 12))
    // Test 2: Batch processing
    val t2 =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0.0f, 192.0f, dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(listOf(4, 3, 4, 4))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val out2 =
        when (val tryResult = t2.upsampleBilinear2d(8, 8, false)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(out2.dims() == listOf(4, 3, 8, 8))
    // Test 3: Asymmetric scale factors
    val t3 =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0.0f, 16.0f, dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(listOf(1, 1, 4, 4))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val out3 =
        when (val tryResult = t3.upsampleBilinear2dWithScale(2.0, 3.0, false)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(out3.dims() == listOf(1, 1, 8, 12))
    // Test 4: Fractional scale factors
    val t4 =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0.0f, 16.0f, dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(listOf(1, 1, 4, 4))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val out4 =
        when (val tryResult = t4.upsampleBilinear2dWithScale(1.5, 1.5, false)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(out4.dims() == listOf(1, 1, 6, 6))
    // Test 5: Single pixel output
    val t5 =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0.0f, 16.0f, dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(listOf(1, 1, 4, 4))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val out5 =
        when (val tryResult = t5.upsampleBilinear2d(1, 1, false)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(out5.dims() == listOf(1, 1, 1, 1))
    val value =
        when (
            val tryResult =
                when (val tryResult = out5.flattenAll()) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.toVec1<Float>()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }[0]
    check(value.isFinite()) { "Single pixel value is not finite" }
    // Test 6: Large scale factor
    val t6 =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0.0f, 4.0f, dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(listOf(1, 1, 2, 2))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val out6 =
        when (val tryResult = t6.upsampleBilinear2dWithScale(5.0, 5.0, false)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(out6.dims() == listOf(1, 1, 10, 10))
    return Result.Ok(Unit)
}

// ============================================================================
// Special Behavior Tests
// ============================================================================
public fun bilinearIdentity(dev: Device): Result<Unit> {
    // Test that upsampling to the same size returns an identical tensor
    val t =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0.0f, 16.0f, dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(listOf(1, 1, 4, 4))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val output =
        when (val tryResult = t.upsampleBilinear2d(4, 4, false)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val diff =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (
                            val tryResult =
                                when (val tryResult = (t - output)) {
                                    is Result.Ok<*> -> tryResult.value
                                    is Result.Err<*> -> return Result.Err(tryResult.error)
                                }.abs()
                        ) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.flattenAll()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.max(0)
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = diff.toVec0<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } < 1e-6,
    )
    return Result.Ok(Unit)
}

public fun bilinearAlignCornersDifference(dev: Device): Result<Unit> {
    // Test that alignCorners parameter produces different results
    val t =
        when (
            val tryResult =
                when (val tryResult = Tensor.arange(0.0f, 16.0f, dev)) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.reshape(listOf(1, 1, 4, 4))
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val outputFalse =
        when (val tryResult = t.upsampleBilinear2d(8, 8, false)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    val outputTrue =
        when (val tryResult = t.upsampleBilinear2d(8, 8, true)) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    // Results should be different between alignCorners modes
    val diff =
        when (
            val tryResult =
                when (
                    val tryResult =
                        when (val tryResult = (outputFalse - outputTrue)) {
                            is Result.Ok<*> -> tryResult.value
                            is Result.Err<*> -> return Result.Err(tryResult.error)
                        }.abs()
                ) {
                    is Result.Ok<*> -> tryResult.value
                    is Result.Err<*> -> return Result.Err(tryResult.error)
                }.sumAll()
        ) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        }
    check(
        when (val tryResult = diff.toVec0<Float>()) {
            is Result.Ok<*> -> tryResult.value
            is Result.Err<*> -> return Result.Err(tryResult.error)
        } > 0.1,
    )
    return Result.Ok(Unit)
}
