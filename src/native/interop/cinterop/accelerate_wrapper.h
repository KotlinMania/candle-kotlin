#ifndef CANDLE_ACCELERATE_ACCELERATE_WRAPPER_H
#define CANDLE_ACCELERATE_ACCELERATE_WRAPPER_H

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

void candleAccelerateSgemmFfi(const char* transa, const char* transb, void* m, void* n, void* k, void* alpha, void* a, void* lda, void* b, void* ldb, void* beta, void* c, void* ldc);
void candleAccelerateDgemmFfi(const char* transa, const char* transb, void* m, void* n, void* k, void* alpha, void* a, void* lda, void* b, void* ldb, void* beta, void* c, void* ldc);
void candleAccelerateVvexpf(void* dst, void* src, void* len);
void candleAccelerateVvexp(void* dst, void* src, void* len);
void candleAccelerateVvsqrtf(void* dst, void* src, void* len);
void candleAccelerateVvsqrt(void* dst, void* src, void* len);
void candleAccelerateVvsinf(void* dst, void* src, void* len);
void candleAccelerateVvsin(void* dst, void* src, void* len);
void candleAccelerateVvcosf(void* dst, void* src, void* len);
void candleAccelerateVvcos(void* dst, void* src, void* len);
void candleAccelerateVvlogf(void* dst, void* src, void* len);
void candleAccelerateVvlog(void* dst, void* src, void* len);
void candleAccelerateVvtanhf(void* dst, void* src, void* len);
void candleAccelerateVvtanh(void* dst, void* src, void* len);
void candleAccelerateVDspVaddD(void* arg0, int64_t arg1, void* arg2, int64_t arg3, void* arg4, int64_t arg5, uint64_t arg6);
void candleAccelerateVDspVadd(void* arg0, int64_t arg1, void* arg2, int64_t arg3, void* arg4, int64_t arg5, uint64_t arg6);
void candleAccelerateVDspVsubD(void* arg0, int64_t arg1, void* arg2, int64_t arg3, void* arg4, int64_t arg5, uint64_t arg6);
void candleAccelerateVDspVsub(void* arg0, int64_t arg1, void* arg2, int64_t arg3, void* arg4, int64_t arg5, uint64_t arg6);
void candleAccelerateVDspVmulD(void* arg0, int64_t arg1, void* arg2, int64_t arg3, void* arg4, int64_t arg5, uint64_t arg6);
void candleAccelerateVDspVmul(void* arg0, int64_t arg1, void* arg2, int64_t arg3, void* arg4, int64_t arg5, uint64_t arg6);
void candleAccelerateVDspVdivD(void* arg0, int64_t arg1, void* arg2, int64_t arg3, void* arg4, int64_t arg5, uint64_t arg6);
void candleAccelerateVDspVdiv(void* arg0, int64_t arg1, void* arg2, int64_t arg3, void* arg4, int64_t arg5, uint64_t arg6);
void candleAccelerateVDspVminD(void* arg0, int64_t arg1, void* arg2, int64_t arg3, void* arg4, int64_t arg5, uint64_t arg6);
void candleAccelerateVDspVmin(void* arg0, int64_t arg1, void* arg2, int64_t arg3, void* arg4, int64_t arg5, uint64_t arg6);
void candleAccelerateVDspVmaxD(void* arg0, int64_t arg1, void* arg2, int64_t arg3, void* arg4, int64_t arg5, uint64_t arg6);
void candleAccelerateVDspVmax(void* arg0, int64_t arg1, void* arg2, int64_t arg3, void* arg4, int64_t arg5, uint64_t arg6);

#ifdef __cplusplus
}
#endif

#endif /* CANDLE_ACCELERATE_ACCELERATE_WRAPPER_H */
