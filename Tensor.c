#include "Tensor.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <immintrin.h>
#include <omp.h>
#include <assert.h>
#include <float.h>
#include <math.h>

#define TILE_SIZE 32
typedef float afloat __attribute__ ((__aligned__(256)));

typedef __m256 float8;

#define LoadFloat8(PTR) (_mm256_load_ps(PTR))
#define StoreFloat8(PTR, VAL) *((float8 *)(PTR)) = (VAL);
#define AddFloat8(PTR, VAL) *((float8 *)(PTR)) += (VAL);
#define BroadcastFloat8(VAL) (_mm256_set1_ps(VAL))
#define FmaddFloat8(A, B, C) (_mm256_fmadd_ps((A), (B), (C)))

inline void load_tensor1d(struct Tensor *t, 
    unsigned long d1, 
    unsigned long *offset, 
    char *buffer)
{

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

inline void load_tensor2d(struct Tensor *t, 
    unsigned long d1,
    unsigned long d2, 
    unsigned long *offset, 
    char *buffer) 
{
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

inline void load_tensor3d(struct Tensor *t, 
    unsigned long d1, 
    unsigned long d2, 
    unsigned long d3,
    unsigned long *offset, 
    char *buffer) 
{
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

inline void load_tensor4d(struct Tensor *t, 
    unsigned long d1, 
    unsigned long d2, 
    unsigned long d3,
    unsigned long d4,
    unsigned long *offset, 
    char *buffer) 
{
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

inline struct Tensor create_tensor(unsigned int* shape, unsigned int ndim) 
{
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

inline float random_float()
{
    return (float)rand() / (float)RAND_MAX;
}

inline struct Tensor rand_tensor(unsigned int *shape, unsigned int ndim)
{
  struct Tensor s = create_tensor(shape, ndim);
  
  for (size_t i=0; i<s.size; i++)
    s.data[i] = (float)rand() / (float)RAND_MAX;
  return s;
}

inline void initialize_linear(struct Linear **layer) 
{
  *layer = (struct Linear*)malloc(sizeof(struct Linear));
  (*layer)->w = (struct Tensor*)malloc(sizeof(struct Tensor));
  (*layer)->b = (struct Tensor*)malloc(sizeof(struct Tensor));
}

inline void initialize_ln(struct LayerNorm **ln)  
{
  *ln = (struct LayerNorm*)malloc(sizeof(struct LayerNorm));
  (*ln)->gamma = (struct Tensor*)malloc(sizeof(struct Tensor));
  (*ln)->beta = (struct Tensor*)malloc(sizeof(struct Tensor));
}

inline void load_linear(struct Linear *layer, unsigned int d1, unsigned int d2, unsigned long *offset, char *buffer) 
{
  load_tensor2d(layer->w, d1, d2, offset, buffer);
  load_tensor1d(layer->b, d1, offset, buffer);
}

inline void load_ln(struct LayerNorm *layer, unsigned int d1, unsigned long *offset, char *buffer) 
{
  load_tensor1d(layer->gamma, d1, offset, buffer);
  load_tensor1d(layer->beta, d1, offset, buffer);
}

inline void free_tensor(struct Tensor *tensor) 
{
    if (tensor != NULL) {
        if (tensor->data != NULL) {
            free(tensor->data); 
        }
        if (tensor->shape != NULL) {
            free(tensor->shape);
        }
    }
}

inline void free_linear(struct Linear *linear) 
{
    if (linear != NULL) {
        free_tensor(linear->w);
        free_tensor(linear->b);
        free(linear);
    }
}

inline void free_ln(struct LayerNorm *ln) 
{
    if (ln != NULL) {
        free_tensor(ln->gamma);
        free_tensor(ln->beta);
        free(ln); 
    }
}

inline void print_tensor_shape(const char* name, const struct Tensor *tensor) 
{
    printf("%s shape: [", name);
    for (unsigned int i = 0; i < tensor->ndim; ++i) {
        printf("%u", tensor->shape[i]);
        if (i < tensor->ndim - 1) {
            printf(", ");
        }
    }
    printf("]\n");
}


void print_tensor(const struct Tensor *tensor) {
    if (tensor->ndim == 1) {
        for (unsigned int i = 0; i < tensor->shape[0]; ++i) {
            if (i >= 5 && i < tensor->shape[0] - 5) {
                if (i == 5) printf("... ");
                continue;
            }
            printf("%f ", tensor->data[i]);
        }

    } else if (tensor->ndim == 2) {
        for (unsigned int i = 0; i < tensor->shape[0]; ++i) {
            if (i >= 5 && i < tensor->shape[0] - 5) {
                if (i == 5) printf("...\n");
                continue;
            }
            for (unsigned int j = 0; j < tensor->shape[1]; ++j) {
                if (j >= 5 && j < tensor->shape[1] - 5) {
                    if (j == 5) printf("... ");
                    continue;
                }
                printf("%f ", tensor->data[i * tensor->shape[1] + j]);
            }
            printf("\n");
        }
    } else if (tensor->ndim == 3) {
        for (unsigned int i = 0; i < tensor->shape[0]; ++i) {
            if (i >= 5 && i < tensor->shape[0] - 5) {
                if (i == 5) printf("...\n");
                continue;
            }
            printf("[");
            for (unsigned int j = 0; j < tensor->shape[1]; ++j) {
                if (j >= 5 && j < tensor->shape[1] - 5) {
                    if (j == 5) printf("... ");
                    continue;
                }
                printf("[");
                for (unsigned int k = 0; k < tensor->shape[2]; ++k) {
                    if (k >= 5 && k < tensor->shape[2] - 5) {
                        if (k == 5) printf("... ");
                        continue;
                    }
                    printf("%f ", tensor->data[i * tensor->shape[1] * tensor->shape[2] + j * tensor->shape[2] + k]);
                }
                printf("]\n");
            }
            printf("]\n");
        }
    } else if (tensor->ndim == 4) {
        for (unsigned int i = 0; i < tensor->shape[0]; ++i) {
            if (i >= 5 && i < tensor->shape[0] - 5) {
                if (i == 5) printf("...\n");
                continue;
            }
            printf("[");
            for (unsigned int j = 0; j < tensor->shape[1]; ++j) {
                if (j >= 5 && j < tensor->shape[1] - 5) {
                    if (j == 5) printf("... ");
                    continue;
                }
                printf("[");
                for (unsigned int k = 0; k < tensor->shape[2]; ++k) {
                    if (k >= 5 && k < tensor->shape[2] - 5) {
                        if (k == 5) printf("... ");
                        continue;
                    }
                    printf("[");
                    for (unsigned int l = 0; l < tensor->shape[3]; ++l) {
                        if (l >= 5 && l < tensor->shape[3] - 5) {
                            if (l == 5) printf("... ");
                            continue;
                        }
                        printf("%f ", tensor->data[i * tensor->shape[1] * tensor->shape[2] * tensor->shape[3] + j * tensor->shape[2] * tensor->shape[3] + k * tensor->shape[3] + l]);
                    }
                    printf("]");
                }
                printf("]");
            }
            printf("]\n");
        }
    } else {
        printf("Printing for tensors with ndim > 4 is not implemented.\n");
    }
}
inline void arange(int **arr, int low, int high) 
{
  *arr = (int *)malloc(sizeof(int) * abs(high - low));
  size_t size = abs(high-low);
  for (int i=0; i<size; i++) {
    (*arr)[i] = low + i;
  }
}

inline void arr_zeros(int **arr, unsigned int size)  
{
   *arr = (int *)calloc(sizeof(int), size);
}
  
inline void arr_ones(int **arr, unsigned int size) 
{
  *arr = (int *)malloc(sizeof(int) * size);
  for (int i=0; i<size; i++) {
    (*arr)[i] = 1;
  }
}


inline void print_first_elements(struct Tensor *t) 
{
  for (int i=0; i<5; i++) {
    printf("%f ", t->data[i]);
  }
  printf("\n");
}

inline void map_embeddings(struct Tensor *t1, 
              struct Tensor *t2, 
              int hidden_size,
              int *tokens, 
              int n_tokens) 
{
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

inline void mm_f32(struct Tensor *a, struct Tensor *b, struct Tensor *c) 
{
    int a_rows = a->shape[0];
    int a_cols = a->shape[1];
    int b_rows = b->shape[0];
    int b_cols = b->shape[1];

    // Assert that the number of columns in A matches the number of rows in B
    assert(a_cols == b_rows && "Number of columns in A must match the number of rows in B");

    int i, j, k, i1, j1, k1;

    #pragma omp parallel for private(i, j, k, i1, j1, k1) shared(a, b, c, a_rows, a_cols, b_cols) schedule(dynamic)
    for (i = 0; i < a_rows; i += TILE_SIZE) {
        for (j = 0; j < b_cols; j += TILE_SIZE) {
            for (k = 0; k < a_cols; k += TILE_SIZE) {
                // Process a block/tile
                for (i1 = i; i1 < i + TILE_SIZE && i1 < a_rows; i1++) {
                    for (j1 = j; j1 < j + TILE_SIZE && j1 < b_cols; j1++) {
                        float sum = 0.0f;
                        for (k1 = k; k1 < k + TILE_SIZE && k1 < a_cols; k1++) {
                            sum += a->data[i1 * a_cols + k1] * b->data[k1 * b_cols + j1];
                        }
                        #pragma omp atomic
                        c->data[i1 * b_cols + j1] += sum;
                    }
                }
            }
        }
    }
}

void sum_tensors(const struct Tensor* a, const struct Tensor* b, struct Tensor* result) 
{
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

void _sum_tensors(struct Tensor *a, const struct Tensor *b) 
{
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

void sub_tensors(const struct Tensor* a, const struct Tensor* b, struct Tensor* result) 
{
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

void _sub_tensors(struct Tensor *a, const struct Tensor *b) 
{
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


void mul_tensors(const struct Tensor* a, const struct Tensor* b, struct Tensor* result) 
{
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

void _mul_tensors(struct Tensor *a, const struct Tensor *b) 
{
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


inline void div_tensors(const struct Tensor* a, const struct Tensor* b, struct Tensor* result) 
{
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

inline void _div_tensors(struct Tensor *a, const struct Tensor *b) 
{
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

inline float reduce_sum(struct Tensor *a)
{
  float s = 0.0f;
  for (size_t i=0; i<a->size; i++)
    s += a->data[i];
  return s;
}

float tensor_max(const struct Tensor *tensor) 
{
    if (tensor->size == 0) {
        printf("Error: Tensor is empty.\n");
        return 0.0f;
    }

    float max_val = tensor->data[0];
    #pragma omp parallel for reduction(max:max_val)
    for (unsigned long i = 1; i < tensor->size; ++i) {
        if (tensor->data[i] > max_val) {
            max_val = tensor->data[i];
        }
    }
    return max_val;
}

float tensor_min(const struct Tensor *tensor) 
{
    if (tensor->size == 0) {
        printf("Error: Tensor is empty.\n");
        return 0.0f;
    }

    float min_val = tensor->data[0];
    #pragma omp parallel for reduction(min:min_val)
    for (unsigned long i = 1; i < tensor->size; ++i) {
        if (tensor->data[i] < min_val) {
            min_val = tensor->data[i];
        }
    }
    return min_val;
}

struct Tensor reduce_sum_axis(const struct Tensor* a, unsigned int axis) 
{
    struct Tensor result;
    if (axis >= a->ndim) {
        printf("Error: Axis %u is out of bounds for tensor with %u dimensions.\n", axis, a->ndim);
        result.ndim = 0;
        result.size = 0;
        result.data = NULL;
        result.shape = NULL;
        return result;
    }

    // Calculate the shape of the result tensor
    result.ndim = a->ndim - 1;
    result.shape = (unsigned int*)malloc(result.ndim * sizeof(unsigned int));
    result.size = 1;
    for (unsigned int i = 0, j = 0; i < a->ndim; ++i) {
        if (i != axis) {
            result.shape[j++] = a->shape[i];
            result.size *= a->shape[i];
        }
    }
    result.data = (float*)malloc(result.size * sizeof(float));

    // Initialize result tensor to zero
    for (unsigned long i = 0; i < result.size; ++i) {
        result.data[i] = 0.0f;
    }

    // Perform the reduction sum along the specified axis
    #pragma omp parallel for
    for (unsigned long i = 0; i < a->size; ++i) {
        int idx[4] = {0};  // Supports up to 4D tensors for now
        int temp = i;
        for (int d = a->ndim - 1; d >= 0; --d) {
            idx[d] = temp % a->shape[d];
            temp /= a->shape[d];
        }

        int result_idx = 0;
        for (unsigned int d = 0; d < a->ndim; ++d) {
            if (d != axis) {
                result_idx = result_idx * a->shape[d] + idx[d];
            }
        }

        #pragma omp atomic
        result.data[result_idx] += a->data[i];
    }

    return result;
}

struct Tensor reduce_max_axis(const struct Tensor* a, unsigned int axis) 
{
    struct Tensor result;
    if (axis >= a->ndim) {
        printf("Error: Axis %u is out of bounds for tensor with %u dimensions.\n", axis, a->ndim);
        result.ndim = 0;
        result.size = 0;
        result.data = NULL;
        result.shape = NULL;
        return result;
    }

    // Calculate the shape of the result tensor
    result.ndim = a->ndim - 1;
    result.shape = (unsigned int*)malloc(result.ndim * sizeof(unsigned int));
    result.size = 1;
    for (unsigned int i = 0, j = 0; i < a->ndim; ++i) {
        if (i != axis) {
            result.shape[j++] = a->shape[i];
            result.size *= a->shape[i];
        }
    }
    result.data = (float*)malloc(result.size * sizeof(float));

    // Initialize result tensor to -INFINITY
    for (unsigned long i = 0; i < result.size; ++i) {
        result.data[i] = FLT_MIN;
    }

    // Perform the reduction max along the specified axis
    #pragma omp parallel for
    for (unsigned long i = 0; i < a->size; ++i) {
        int idx[4] = {0};  // Supports up to 4D tensors for now
        int temp = i;
        for (int d = a->ndim - 1; d >= 0; --d) {
            idx[d] = temp % a->shape[d];
            temp /= a->shape[d];
        }

        int result_idx = 0;
        for (unsigned int d = 0; d < a->ndim; ++d) {
            if (d != axis) {
                result_idx = result_idx * a->shape[d] + idx[d];
            }
        }

        #pragma omp critical
        if (a->data[i] > result.data[result_idx]) {
            result.data[result_idx] = a->data[i];
        }
    }

    return result;
}

struct Tensor reduce_min_axis(const struct Tensor* a, unsigned int axis) 
{
    struct Tensor result;
    if (axis >= a->ndim) {
        printf("Error: Axis %u is out of bounds for tensor with %u dimensions.\n", axis, a->ndim);
        result.ndim = 0;
        result.size = 0;
        result.data = NULL;
        result.shape = NULL;
        return result;
    }

    // Calculate the shape of the result tensor
    result.ndim = a->ndim - 1;
    result.shape = (unsigned int*)malloc(result.ndim * sizeof(unsigned int));
    result.size = 1;
    for (unsigned int i = 0, j = 0; i < a->ndim; ++i) {
        if (i != axis) {
            result.shape[j++] = a->shape[i];
            result.size *= a->shape[i];
        }
    }
    result.data = (float*)malloc(result.size * sizeof(float));

    // Initialize result tensor to INFINITY
    for (unsigned long i = 0; i < result.size; ++i) {
        result.data[i] = FLT_MAX;
    }

    // Perform the reduction min along the specified axis
    #pragma omp parallel for
    for (unsigned long i = 0; i < a->size; ++i) {
        int idx[4] = {0};  // Supports up to 4D tensors for now
        int temp = i;
        for (int d = a->ndim - 1; d >= 0; --d) {
            idx[d] = temp % a->shape[d];
            temp /= a->shape[d];
        }

        int result_idx = 0;
        for (unsigned int d = 0; d < a->ndim; ++d) {
            if (d != axis) {
                result_idx = result_idx * a->shape[d] + idx[d];
            }
        }

        #pragma omp critical
        if (a->data[i] < result.data[result_idx]) {
            result.data[result_idx] = a->data[i];
        }
    }

    return result;
}

float tensor_mean(const struct Tensor* a) 
{
    if (a->size == 0) {
        printf("Error: Tensor is empty.\n");
        return 0.0f;
    }
    float sum = 0.0f;
    #pragma omp parallel for reduction(+:sum)
    for (unsigned long i = 0; i < a->size; ++i) {
        sum += a->data[i];
    }
    return sum / a->size;
}

struct Tensor reduce_mean_axis(const struct Tensor* a, unsigned int axis) 
{
    struct Tensor sum_tensor = reduce_sum_axis(a, axis);
    if (sum_tensor.size == 0) {
        return sum_tensor;
    }

    unsigned int axis_size = a->shape[axis];
    #pragma omp parallel for
    for (unsigned long i = 0; i < sum_tensor.size; ++i) {
        sum_tensor.data[i] /= axis_size;
    }

    return sum_tensor;
}

float tensor_std(const struct Tensor* a) 
{
    if (a->size == 0) {
        printf("Error: Tensor is empty.\n");
        return 0.0f;
    }
    float mean = tensor_mean(a);
    float sum_sq_diff = 0.0f;
    #pragma omp parallel for reduction(+:sum_sq_diff)
    for (unsigned long i = 0; i < a->size; ++i) {
        float diff = a->data[i] - mean;
        sum_sq_diff += diff * diff;
    }
    return sqrtf(sum_sq_diff / a->size);
}

struct Tensor reduce_std_axis(const struct Tensor* a, unsigned int axis) 
{
    struct Tensor mean_tensor = reduce_mean_axis(a, axis);
    struct Tensor std_tensor = reduce_sum_axis(a, axis);

    if (std_tensor.size == 0) {
        return std_tensor;
    }

    unsigned int axis_size = a->shape[axis];
    #pragma omp parallel for
    for (unsigned long i = 0; i < std_tensor.size; ++i) {
        std_tensor.data[i] = 0.0f;
    }

    #pragma omp parallel for
    for (unsigned long i = 0; i < a->size; ++i) {
        int idx[4] = {0};  // Supports up to 4D tensors for now
        int temp = i;
        for (int d = a->ndim - 1; d >= 0; --d) {
            idx[d] = temp % a->shape[d];
            temp /= a->shape[d];
        }

        int result_idx = 0;
        for (unsigned int d = 0; d < a->ndim; ++d) {
            if (d != axis) {
                result_idx = result_idx * a->shape[d] + idx[d];
            }
        }

        float diff = a->data[i] - mean_tensor.data[result_idx];
        #pragma omp atomic
        std_tensor.data[result_idx] += diff * diff;
    }

    #pragma omp parallel for
    for (unsigned long i = 0; i < std_tensor.size; ++i) {
        std_tensor.data[i] = sqrtf(std_tensor.data[i] / axis_size);
    }

    free_tensor(&mean_tensor);
    return std_tensor;
}

void _add_tensor_scalar(struct Tensor* tensor, float scalar) 
{
    #pragma omp parallel for
    for (unsigned long i = 0; i < tensor->size; ++i) 
        tensor->data[i] += scalar;
}

void _mul_tensor_scalar(struct Tensor* tensor, float scalar) 
{
    #pragma omp parallel for
    for (unsigned long i = 0; i < tensor->size; ++i) 
        tensor->data[i] *= scalar;
}
