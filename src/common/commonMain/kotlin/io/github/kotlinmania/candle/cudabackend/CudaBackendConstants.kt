package io.github.kotlinmania.candle.cudabackend

import io.github.kotlinmania.rustlang.std.sync.Mutex
import io.github.kotlinmania.rustlang.std.sync.OnceLock

public var CUDA_PARAM_CACHE: OnceLock<Mutex<ParamCache>> = OnceLock.new()

// Default for the reduced precision setting is false, similar to pytorch.
// https://github.com/pytorch/pytorch/issues/123157
public var MM_F16_REDUCED_PRECISION: AtomicBool = AtomicBool.new(false)

public var MM_BF16_REDUCED_PRECISION: AtomicBool = AtomicBool.new(false)

public var MM_F32_REDUCED_PRECISION: AtomicBool = AtomicBool.new(false)
