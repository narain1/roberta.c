#ifndef HELPER_H
#define HELPER_H
#include <stdalign.h>

#define FLT_MAX 3.402823466e+38F /* max value */
#define FLT_MIN 1.175494351e-38F /* min positive value */

struct Tensor {
  alignas(32) float *data;
  unsigned int *shape;
  unsigned int ndim;
  unsigned long size;
};

struct Linear {
  struct Tensor *w;
  struct Tensor *b;
};

struct LayerNorm {
  struct Tensor *gamma;
  struct Tensor *beta;
};

void load_tensor1d(
    struct Tensor *t,
    unsigned long d1,
    unsigned long *offset,
    char *buffer);

void load_tensor2d(
    struct Tensor *t,
    unsigned long d1,
    unsigned long d2,
    unsigned long *offset,
    char *buffer);

void load_tensor3d(
    struct Tensor *t,
    unsigned long d1,
    unsigned long d2,
    unsigned long d3,
    unsigned long *offset,
    char *buffer);

void load_tensor4d(
    struct Tensor *t,
    unsigned long d1,
    unsigned long d2,
unsigned long d3,
    unsigned long d4,
    unsigned long *offset,
    char *buffer);

void initialize_linear(struct Linear **layer);

void initialize_ln(struct LayerNorm **ln);

void load_linear(
    struct Linear *layer,
    unsigned int d1,
    unsigned int d2,
    unsigned long *offset,
    char *buffer);

void load_ln(
    struct LayerNorm *layer,
    unsigned int d1,
    unsigned long *offset,
    char *buffer);

void free_tensor(struct Tensor *tensor);

void free_linear(struct Linear *layer);

void free_ln(struct LayerNorm *ln);

void print_tensor_shape(const char *name, const struct Tensor *tensor);

void print_first_elements(struct Tensor *t);

void print_tensor(const struct Tensor *t);

void arange(int **arr, int low, int high);

void arr_zeros(int **arr, unsigned int size);

void arr_ones(int **arr, unsigned int size);

void map_embeddings(
    struct Tensor *t1,
    struct Tensor *t2,
    int hidden_size,
    int *tokens,
    int n_tokens);

struct Tensor create_tensor(unsigned int *shape, unsigned int ndim);
struct Tensor rand_tensor(unsigned int *shape, unsigned int ndim);

void sum_tensors(const struct Tensor *a, const struct Tensor *b, struct Tensor *result);
void _sum_tensors(struct Tensor *a, const struct Tensor *b);
void sum_tensors_broadcast(const struct Tensor *a, const struct Tensor *b, struct Tensor *result);
void _sum_tensors_broadcast(struct Tensor *a, const struct Tensor *b);
void _sum_tensor_scaler(struct Tensor *a, float scaler);
    
void sub_tensors(const struct Tensor* a, const struct Tensor* b, struct Tensor* result);
void _sub_tensors(struct Tensor *a, const struct Tensor *b);

void mul_tensors(const struct Tensor* a, const struct Tensor* b, struct Tensor* result);
void _mul_tensors(struct Tensor *a, const struct Tensor *b);
void _mul_tensor_scaler(struct Tensor *a, float scaler);

void div_tensors(const struct Tensor* a, const struct Tensor* b, struct Tensor* result);
void _div_tensors(struct Tensor *a, const struct Tensor *b);

float reduce_sum(struct Tensor *t);
struct Tensor reduce_sum_axis(const struct Tensor *a, unsigned int axis);

float tensor_max(const struct Tensor *t);
struct Tensor reduce_max_axis(const struct Tensor *a, unsigned int axis);

float tensor_min(const struct Tensor *t);
struct Tensor reduce_min_axis(const struct Tensor *a, unsigned int axis);

float tensor_mean(const struct Tensor *a);
struct Tensor reduce_mean_axis(const  struct Tensor *a, unsigned int axis); 

float tensor_std(const struct Tensor *a);
struct Tensor reduce_std_axis(const struct Tensor *a, unsigned int axis);

void mm_f32(const struct Tensor *a, const struct Tensor *b, struct Tensor *c);

void transpose_tensor(const struct Tensor *inp, struct Tensor *out);
#endif // HELPER_H
