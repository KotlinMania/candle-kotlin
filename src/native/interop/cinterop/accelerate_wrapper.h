#ifndef CANDLE_ACCELERATE_ACCELERATE_WRAPPER_H
#define CANDLE_ACCELERATE_ACCELERATE_WRAPPER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void candle_accelerate_sgemmFfi(const char* transa, const char* transb, void* m, void* n, void* k, void* alpha, void* a, void* lda, void* b, void* ldb, void* beta, void* c, void* ldc);
void candle_accelerate_dgemmFfi(const char* transa, const char* transb, void* m, void* n, void* k, void* alpha, void* a, void* lda, void* b, void* ldb, void* beta, void* c, void* ldc);
void candle_accelerate_vvexpf(void* dst, void* src, void* len);
void candle_accelerate_vvexp(void* dst, void* src, void* len);
void candle_accelerate_vvsqrtf(void* dst, void* src, void* len);
void candle_accelerate_vvsqrt(void* dst, void* src, void* len);
void candle_accelerate_vvsinf(void* dst, void* src, void* len);
void candle_accelerate_vvsin(void* dst, void* src, void* len);
void candle_accelerate_vvcosf(void* dst, void* src, void* len);
void candle_accelerate_vvcos(void* dst, void* src, void* len);
void candle_accelerate_vvlogf(void* dst, void* src, void* len);
void candle_accelerate_vvlog(void* dst, void* src, void* len);
void candle_accelerate_vvtanhf(void* dst, void* src, void* len);
void candle_accelerate_vvtanh(void* dst, void* src, void* len);
void candle_accelerate_vDspVaddD(void* arg0, int64_t arg1, void* arg2, int64_t arg3, void* arg4, int64_t arg5, uint64_t arg6);
void candle_accelerate_vDspVadd(void* arg0, int64_t arg1, void* arg2, int64_t arg3, void* arg4, int64_t arg5, uint64_t arg6);
void candle_accelerate_vDspVsubD(void* arg0, int64_t arg1, void* arg2, int64_t arg3, void* arg4, int64_t arg5, uint64_t arg6);
void candle_accelerate_vDspVsub(void* arg0, int64_t arg1, void* arg2, int64_t arg3, void* arg4, int64_t arg5, uint64_t arg6);
void candle_accelerate_vDspVmulD(void* arg0, int64_t arg1, void* arg2, int64_t arg3, void* arg4, int64_t arg5, uint64_t arg6);
void candle_accelerate_vDspVmul(void* arg0, int64_t arg1, void* arg2, int64_t arg3, void* arg4, int64_t arg5, uint64_t arg6);
void candle_accelerate_vDspVdivD(void* arg0, int64_t arg1, void* arg2, int64_t arg3, void* arg4, int64_t arg5, uint64_t arg6);
void candle_accelerate_vDspVdiv(void* arg0, int64_t arg1, void* arg2, int64_t arg3, void* arg4, int64_t arg5, uint64_t arg6);
void candle_accelerate_vDspVminD(void* arg0, int64_t arg1, void* arg2, int64_t arg3, void* arg4, int64_t arg5, uint64_t arg6);
void candle_accelerate_vDspVmin(void* arg0, int64_t arg1, void* arg2, int64_t arg3, void* arg4, int64_t arg5, uint64_t arg6);
void candle_accelerate_vDspVmaxD(void* arg0, int64_t arg1, void* arg2, int64_t arg3, void* arg4, int64_t arg5, uint64_t arg6);
void candle_accelerate_vDspVmax(void* arg0, int64_t arg1, void* arg2, int64_t arg3, void* arg4, int64_t arg5, uint64_t arg6);

#ifdef __cplusplus
}
#endif

#endif /* CANDLE_ACCELERATE_ACCELERATE_WRAPPER_H */
