// port-lint: source dummy_dtype.rs
package io.github.kotlinmania.candle

import io.github.kotlinmania.anyhow.Ok
import io.github.kotlinmania.anyhow.Result

/**
 * Dummy data types for experimental/future float formats
 *
 * These are placeholder types for experimental floating-point formats
 * that are defined in the safetensors spec but not yet fully implemented.
 */

/**
 * 6-bit float with 2 exponent bits and 3 mantissa bits (MX6 format)
 * This is a dummy type.
 */
public class F6E2M3 {
    public fun fmt(f: StringBuilder): Result<Unit> {
        val write = { out: StringBuilder, text: String -> out.append(text); Ok(Unit) }
        return write(f, "F6E2M3")
    }
}

/**
 * 6-bit float with 3 exponent bits and 2 mantissa bits (MX6 format)
 * This is a dummy type.
 */
public class F6E3M2 {
    public fun fmt(f: StringBuilder): Result<Unit> {
        val write = { out: StringBuilder, text: String -> out.append(text); Ok(Unit) }
        return write(f, "F6E3M2")
    }
}

/**
 * 4-bit float (MX4 format)
 * This is a dummy type.
 */
public class F4 {
    public fun fmt(f: StringBuilder): Result<Unit> {
        val write = { out: StringBuilder, text: String -> out.append(text); Ok(Unit) }
        return write(f, "F4")
    }
}

/**
 * 8-bit float with 8 exponent bits and 0 mantissa bits
 * This is a dummy type.
 */
public class F8E8M0 {
    public fun fmt(f: StringBuilder): Result<Unit> {
        val write = { out: StringBuilder, text: String -> out.append(text); Ok(Unit) }
        return write(f, "F8E8M0")
    }
}
