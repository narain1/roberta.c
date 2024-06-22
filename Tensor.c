#include "Tensor.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>


typedef float afloat __attribute__ ((__aligned__(256)));

typedef __m256 float8;

#define LoadFloat8(PTR) (_mm256_load_ps(PTR))
#define StoreFloat8(PTR, VAL) *((float8 *)(PTR)) = (VAL);
#define AddFloat8(PTR, VAL) *((float8 *)(PTR)) += (VAL);
#define BroadcastFloat8(VAL) (_mm256_set1_ps(VAL))
#define FmaddFloat8(A, B, C) (_mm256_fmadd_ps((A), (B), (C)))


void load_tensor1d(struct Tensor *t, 
    unsigned long d1, 
    unsigned long *offset, 
    char *buffer) {

  size_t dim = (size_t)d1;
  float *data = (float*)malloc(sizeof(float)*d1);
  unsigned int  *s = (unsigned int*)malloc(sizeof(unsigned int));
  *s = d1;
  
  for (size_t i=0; i<dim; i++) {
    data[i] = *((float *)(buffer + *offset + i * sizeof(float)));
  }
  *offset += dim * sizeof(float);
  t->data = data;
  t->shape = s;
  t->ndim = 1;
  t->size = dim;
}


void load_tensor2d(struct Tensor *t, 
    unsigned long d1,
    unsigned long d2, 
    unsigned long *offset, 
    char *buffer) {
  size_t dim = d1 * d2;
  float *data = (float*)malloc(sizeof(float)*dim);
  unsigned int *s = (unsigned int*)malloc(sizeof(unsigned int) * 2);

  s[0] = d1;
  s[1] = d2;
  
  for (size_t i=0; i<dim; i++) {
    data[i] = *((float *)(buffer + *offset + i * sizeof(float)));
  }
  *offset += dim * sizeof(float);
  t->data = data;
  t->shape = s;
  t->ndim = 2;
  t->size = dim;
}

void load_tensor3d(struct Tensor *t, 
    unsigned long d1, 
    unsigned long d2, 
    unsigned long d3,
    unsigned long *offset, 
    char *buffer) {
  size_t dim = d1 * d2 * d3;
  float *data = (float*)malloc(sizeof(float)*dim);

  unsigned int *s = (unsigned int*)malloc(sizeof(unsigned int) * 2);
  s[0] = d1;
  s[1] = d2;
  s[2] = d3;

  for (size_t i=0; i<dim; i++) {
    data[i] = *((float *)(buffer + *offset + i * sizeof(float)));
  }
  *offset += dim * sizeof(float);
  t->data = data;
  t->shape = s;
  t->ndim = 3;
  t->size = dim;
}

void load_tensor4d(struct Tensor *t, 
    unsigned long d1, 
    unsigned long d2, 
    unsigned long d3,
    unsigned long d4,
    unsigned long *offset, 
    char *buffer) {
  size_t dim = d1 * d2;
  float *data = (float*)malloc(sizeof(float)*dim);

  unsigned int *s = (unsigned int*)malloc(sizeof(unsigned int) * 2);
  s[0] = d1;
  s[1] = d2;
  s[2] = d3;
  s[3] = d4;

  for (size_t i=0; i<dim; i++) {
    data[i] = *((float *)(buffer + *offset + i * sizeof(float)));
  }
  *offset += dim * sizeof(float);
  t->data = data;
  t->shape = s;
  t->ndim = 4;
  t->size = dim;
}

struct Tensor create_tensor(unsigned int* shape, unsigned int ndim) {
    struct Tensor t;
    t.shape = (unsigned int*)malloc(sizeof(unsigned int) * ndim);
    memcpy(t.shape, shape, sizeof(unsigned int) * ndim);
    t.ndim = ndim;
    t.size = 1;
    for (unsigned int i = 0; i < ndim; ++i) {
        t.size *= shape[i];
    }
    t.data = (float*)malloc(t.size * sizeof(float)); // Ensure alignment for AVX
    return t;
}


void initialize_linear(struct Linear **layer) {
  *layer = (struct Linear*)malloc(sizeof(struct Linear));
  (*layer)->w = (struct Tensor*)malloc(sizeof(struct Tensor));
  (*layer)->b = (struct Tensor*)malloc(sizeof(struct Tensor));
}

void initialize_ln(struct LayerNorm **ln) {
  *ln = (struct LayerNorm*)malloc(sizeof(struct LayerNorm));
  (*ln)->gamma = (struct Tensor*)malloc(sizeof(struct Tensor));
  (*ln)->beta = (struct Tensor*)malloc(sizeof(struct Tensor));
}

void load_linear(struct Linear *layer, unsigned int d1, unsigned int d2, unsigned long *offset, char *buffer) {
  load_tensor2d(layer->w, d1, d2, offset, buffer);
  load_tensor1d(layer->b, d1, offset, buffer);
}

void load_ln(struct LayerNorm *layer, unsigned int d1, unsigned long *offset, char *buffer) {
  load_tensor1d(layer->gamma, d1, offset, buffer);
  load_tensor1d(layer->beta, d1, offset, buffer);
}


void free_tensor(struct Tensor *tensor) {
    if (tensor != NULL) {
        if (tensor->data != NULL) {
            free(tensor->data); 
        }
        if (tensor->shape != NULL) {
            free(tensor->shape);
        }
        free(tensor); 
    }
}

void free_linear(struct Linear *linear) {
    if (linear != NULL) {
        free_tensor(linear->w);
        free_tensor(linear->b);
        free(linear);
    }
}

void free_ln(struct LayerNorm *ln) {
    if (ln != NULL) {
        free_tensor(ln->gamma);
        free_tensor(ln->beta);
        free(ln); 
    }
}


void print_tensor_shape(const char* name, const struct Tensor *tensor) {
    printf("%s shape: [", name);
    for (unsigned int i = 0; i < tensor->ndim; ++i) {
        printf("%u", tensor->shape[i]);
        if (i < tensor->ndim - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}

void arange(int **arr, int low, int high) {
  *arr = (int *)malloc(sizeof(int) * abs(high - low));
  size_t size = abs(high-low);
  for (int i=0; i<size; i++) {
    (*arr)[i] = low + i;
  }
}

void arr_zeros(int **arr, unsigned int size) {
   *arr = (int *)calloc(sizeof(int), size);
}
  
void arr_ones(int **arr, unsigned int size) {
  *arr = (int *)malloc(sizeof(int) * size);
  for (int i=0; i<size; i++) {
    (*arr)[i] = 1;
  }
}


void print_first_elements(struct Tensor *t) {
  for (int i=0; i<5; i++) {
    printf("%f ", t->data[i]);
  }
  printf("\n");
}

void map_embeddings(struct Tensor *t1, 
              struct Tensor *t2, 
              int hidden_size,
              int *tokens, 
              int n_tokens) {
  size_t dim = n_tokens * hidden_size;
  float *data = (float *)malloc(sizeof(float) * dim);
  unsigned int *s = (unsigned int*)malloc(sizeof(unsigned int) * 2);

  s[0] = hidden_size;
  s[1] = n_tokens;

  for (int i=0; i<n_tokens; i++) {
    memcpy(data+(hidden_size * i), t2->data+(hidden_size * tokens[i]), hidden_size * sizeof(float));
  }
  t1->data = data;
  t1->shape = s;
  t1->ndim = 2;
  t1->size = dim;
}

inline void matmul_dot_inner(
    unsigned regsA, unsigned regsB,
    const afloat * __restrict__ A,
    const afloat * __restrict__ B,
    const afloat *C,
    const int M, const int N, const int K,
    const int m, const int n) {

  float8 csum[regsA][regsB]; // Variable Length Array (VLA) in C99
  for (unsigned i = 0; i < regsA; ++i) {
      for (unsigned j = 0; j < regsB; ++j) {
          csum[i][j] = BroadcastFloat8(0);
      }
  }

  for (int k=0; k<K; k++) {
    for (unsigned ai=0; ai<regsA; ai++) {
      float8 aa = BroadcastFloat8(A[(m+ai) * K + k]);
      for (unsigned bi =0; bi<regsB; bi++) {
        float8 bb = LoadFloat8(&B[k*N + n + bi * 8]);
        csum[ai][bi] = FmaddFloat8(aa, bb, csum[ai][bi]);
      }
    }
  }

  for (unsigned ai=0; ai<regsA; ai++) {
    for (unsigned bi=0; bi<regsB; bi++) {
      StoreFloat8(&C[(m+ai) * N+n+bi*8], csum[ai][bi]);
    }
  }
}

inline void matmul_dot_inner_block(
    unsigned regsA, unsigned regsB,
    const afloat * __restrict__ A,
    const afloat * __restrict__ B,
    afloat *C,
    const int M, const int N, const int K,
    const int jc, const int nc,
    const int pc, const int kc,
    const int ic, const int mc,
    const int jr, const int nr,
    const int ir, const int mr
    ) {

  float8 csum[regsA][regsB]; // Variable Length Array (VLA) in C99
  for (unsigned i = 0; i < regsA; ++i) {
      for (unsigned j = 0; j < regsB; ++j) {
          csum[i][j] = BroadcastFloat8(0);
      }
  }

  for (int k=0; k<kc; k++) {
    for (unsigned ai=0; ai<regsA; ai++) {
      float8 aa = BroadcastFloat8(A[(ic+ir+ai) * K + pc + k]);
      for (unsigned bi=0; bi<regsB; bi++) {
        float8 bb = LoadFloat8(&B[(pc + k) * N +jc + jr + bi * 8]);
        csum[ai][bi] = FmaddFloat8(aa, bb, csum[ai][bi]);
      }
    }
  }

  for (unsigned ai=0; ai<regsA; ai++) {
    for (unsigned bi=0; bi<regsB; bi++) {
      AddFloat8(&C[(ic+ir+ai) * N + jc + jr + bi * 8], csum[ai][bi]);
    }
  }
}

inline void sgemm_simd_block_parallel(
    const int M,
    const int N,
    const int K,
    const float *A,
    const float *B,
    float *C) {
  const int nc = N;
  const int kc = 240;
  const int mc = 120;
  const int nr = 2 * 8;
  const int mr = 6;

  omp_set_num_threads(8);
  for (int jc=0; jc<N; jc+=nc) {
    for (int pc=0; pc<K; pc+=kc) {
      for (int ic=0; ic<M; ic+=mc) {
        #pragma omp parallel for
        for (int jr=0; jr<nc; jr+=nr) {
          for (int ir=0; ir<mc; ir+=mr) {
            matmul_dot_inner_block(6, 2, A,B,C,M,N,K,jc,nc,pc,kc,ic,mc,jr,nr,ir,mr);
          }
        }
      }
    }
  }
}

void sum_tensors(const struct Tensor* a, const struct Tensor* b, struct Tensor* result) {
    if (a->ndim != b->ndim) {
        printf("Error: Tensors must have the same number of dimensions.\n");
        return;
    }
    for (unsigned int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) {
            printf("Error: Tensor shapes must match.\n");
            return;
        }
    }

    // Initialize result tensor
    result->ndim = a->ndim;
    result->shape = (unsigned int*)malloc(result->ndim * sizeof(unsigned int));
    if (!result->shape) {
        printf("Error: Failed to allocate memory for tensor shape.\n");
        return;
    }
    for (unsigned int i = 0; i < result->ndim; i++) {
        result->shape[i] = a->shape[i]; // Correctly copy the shape values
    }
    result->size = a->size;
    result->data = (float*)malloc(result->size * sizeof(float)); // 32-byte alignment for AVX

    for (unsigned int i = 0; i < result->size; i++) {
        result->data[i] = a->data[i] + b->data[i];
    }
}

void sum_tensors_inplace(struct Tensor *a, const struct Tensor *b) {
    if (a->ndim != b->ndim) {
        printf("Error: Tensors must have the same number of dimensions for in-place operation.\n");
        return;
    }
    for (unsigned int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) {
            printf("Error: Tensor shapes must match for in-place operation.\n");
            print_tensor_shape("a: ", a);
            print_tensor_shape("b: ", b);
            return;
        }
    }

    for (size_t i = 0; i < a->size; i++) {
        a->data[i] += b->data[i];
    }
}

void sub_tensors(const struct Tensor* a, const struct Tensor* b, struct Tensor* result) {
    if (a->ndim != b->ndim) {
        printf("Error: Tensors must have the same number of dimensions.\n");
        return;
    }
    for (unsigned int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) {
            printf("Error: Tensor shapes must match.\n");
            return;
        }
    }

    // Initialize result tensor
    result->ndim = a->ndim;
    result->shape = (unsigned int*)malloc(result->ndim * sizeof(unsigned int));
    if (!result->shape) {
        printf("Error: Failed to allocate memory for tensor shape.\n");
        return;
    }
    for (unsigned int i = 0; i < result->ndim; i++) {
        result->shape[i] = a->shape[i]; // Correctly copy the shape values
    }
    result->size = a->size;
    result->data = (float*)malloc(result->size * sizeof(float)); // 32-byte alignment for AVX

    for (unsigned int i = 0; i < result->size; i++) {
        result->data[i] = a->data[i] - b->data[i];
    }
}

void sub_tensors_inplace(struct Tensor *a, const struct Tensor *b) {
    if (a->ndim != b->ndim) {
        printf("Error: Tensors must have the same number of dimensions for in-place operation.\n");
        return;
    }
    for (unsigned int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) {
            printf("Error: Tensor shapes must match for in-place operation.\n");
            print_tensor_shape("a: ", a);
            print_tensor_shape("b: ", b);
            return;
        }
    }

    for (size_t i = 0; i < a->size; i++) {
        a->data[i] -= b->data[i];
    }
}


void mul_tensors(const struct Tensor* a, const struct Tensor* b, struct Tensor* result) {
    if (a->ndim != b->ndim) {
        printf("Error: Tensors must have the same number of dimensions.\n");
        return;
    }
    for (unsigned int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) {
            printf("Error: Tensor shapes must match.\n");
            return;
        }
    }

    // Initialize result tensor
    result->ndim = a->ndim;
    result->shape = (unsigned int*)malloc(result->ndim * sizeof(unsigned int));
    if (!result->shape) {
        printf("Error: Failed to allocate memory for tensor shape.\n");
        return;
    }
    for (unsigned int i = 0; i < result->ndim; i++) {
        result->shape[i] = a->shape[i]; // Correctly copy the shape values
    }
    result->size = a->size;
    result->data = (float*)malloc(result->size * sizeof(float)); // 32-byte alignment for AVX

    for (unsigned int i = 0; i < result->size; i++) {
        result->data[i] = a->data[i] * b->data[i];
    }
}

void mul_tensors_inplace(struct Tensor *a, const struct Tensor *b) {
    if (a->ndim != b->ndim) {
        printf("Error: Tensors must have the same number of dimensions for in-place operation.\n");
        return;
    }
    for (unsigned int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) {
            printf("Error: Tensor shapes must match for in-place operation.\n");
            print_tensor_shape("a: ", a);
            print_tensor_shape("b: ", b);
            return;
        }
    }

    for (size_t i = 0; i < a->size; i++) {
        a->data[i] *= b->data[i];
    }
}


void div_tensors(const struct Tensor* a, const struct Tensor* b, struct Tensor* result) {
    if (a->ndim != b->ndim) {
        printf("Error: Tensors must have the same number of dimensions.\n");
        return;
    }
    for (unsigned int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) {
            printf("Error: Tensor shapes must match.\n");
            return;
        }
    }

    // Initialize result tensor
    result->ndim = a->ndim;
    result->shape = (unsigned int*)malloc(result->ndim * sizeof(unsigned int));
    if (!result->shape) {
        printf("Error: Failed to allocate memory for tensor shape.\n");
        return;
    }
    for (unsigned int i = 0; i < result->ndim; i++) {
        result->shape[i] = a->shape[i]; // Correctly copy the shape values
    }
    result->size = a->size;
    result->data = (float*)malloc(result->size * sizeof(float)); // 32-byte alignment for AVX

    for (unsigned int i = 0; i < result->size; i++) {
        if (b->data[i] == 0) {
            printf("Error: Division by zero at index %u.\n", i);
            result->data[i] = 0; // Handle division by zero appropriately
        } else {
            result->data[i] = a->data[i] / b->data[i];
        }
    }
}

void div_tensors_inplace(struct Tensor *a, const struct Tensor *b) {
    if (a->ndim != b->ndim) {
        printf("Error: Tensors must have the same number of dimensions for in-place operation.\n");
        return;
    }
    for (unsigned int i = 0; i < a->ndim; i++) {
        if (a->shape[i] != b->shape[i]) {
            printf("Error: Tensor shapes must match for in-place operation.\n");
            print_tensor_shape("a: ", a);
            print_tensor_shape("b: ", b);
            return;
        }
    }

    for (size_t i = 0; i < a->size; i++) {
        if (b->data[i] == 0) {
            printf("Error: Division by zero at index %zu.\n", i);
            a->data[i] = 0; // Handle division by zero appropriately
        } else {
            a->data[i] /= b->data[i];
        }
    }
}

