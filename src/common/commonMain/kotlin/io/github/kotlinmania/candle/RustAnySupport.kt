package io.github.kotlinmania.candle

public data class RustTypeId(
    public val kClass: kotlin.reflect.KClass<*>,
    public val tag: String,
) {
    public companion object {
        public inline fun <reified T> of(): RustTypeId {
            val resolvedTag =
                when (T::class) {
                    Byte::class -> "i8"
                    Short::class -> "i16"
                    Int::class -> "i32"
                    Long::class -> "i64"
                    UByte::class -> "u8"
                    UShort::class -> "u16"
                    UInt::class -> "u32"
                    ULong::class -> "u64"
                    Float::class -> "f32"
                    Double::class -> "f64"
                    Boolean::class -> "bool"
                    Char::class -> "char"
                    String::class -> "String"
                    Unit::class -> "()"
                    else -> T::class.simpleName ?: T::class.toString()
                }
            return RustTypeId(T::class, resolvedTag)
        }
    }
}

public class RustAny(
    public val typeId: RustTypeId,
    public val rawValue: Any?,
) {
    public inline fun <reified T> isType(): Boolean = typeId == RustTypeId.of<T>()

    public inline fun <reified T> downcastRef(): T? =
        if (typeId == RustTypeId.of<T>()) rawValue as T else null

    public inline fun <reified T> downcastMut(): T? =
        if (typeId == RustTypeId.of<T>()) rawValue as T else null

    public inline fun <reified T> downcast(): Result<T> =
        if (typeId == RustTypeId.of<T>()) {
            Result.success(rawValue as T)
        } else {
            Result.failure(ClassCastException("RustAny(${typeId.tag}) cannot downcast to ${RustTypeId.of<T>().tag}"))
        }

    public companion object {
        public inline fun <reified T> from(value: T): RustAny =
            RustAny(RustTypeId.of<T>(), value)
    }
}
