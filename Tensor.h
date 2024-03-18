#ifndef HELPER_H
#define HELPER_H
#include <stdalign.h>

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

void print_tensor_shape(const char *name, struct Tensor *tensor);

void print_first_elements(struct Tensor *t);

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

void sum_tensors(
    const struct Tensor *a,
    const struct Tensor *b,
    struct Tensor *result);

void sum_tensors_inplace(
    struct Tensor *a,
    struct Tensor *b);
    
#endif // HELPER_H
