#include "Tensor.h"
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

void reshape(struct Tensor *a, unsigned int *s, int ndim)
{
  size_t reshape_size = 1;
  for (int i=0; i<ndim; i++)
    reshape_size *= s[i];
  if (reshape_size != a->size)
    fprintf(stderr, "reshape tensor size mismatch %zu and %zu", a->size, reshape_size); 
  if (a->ndim != ndim)
    a->shape = realloc(a->shape, ndim);
  for (int i=0; i<ndim; i++)
    a->shape[i] = s[i];
  a->ndim = ndim;
}

void broadcast_sum(struct Tensor *a, struct Tensor *b) 
{
  if (!broadcast_check(a, b)) {
    fprintf(stderr, "Tensors cannot be broadcasted\n");
  }
  for (size_t i=0; i<a->size; i++)
    a->data[i] += b->data[i%b->size];
}

int main() {
  struct Tensor a = arange(10);
  print_tensor(&a);

  printf("attempting reshaping\n");

  unsigned int *s = (unsigned int*)malloc(sizeof(unsigned int) * 2);
  s[0] = 2;
  s[1] = 5;
  reshape(&a, s, 2);
  print_tensor(&a);

  s[0] = 5;
  s[1] = 2;
  reshape(&a, s, 2);
  print_tensor(&a);

  struct Tensor b = arange(27);
  unsigned int *s1 = (unsigned int*)malloc(sizeof(unsigned int) * 3);
  s1[0] = 3;
  s1[1] = 3;
  s1[2] = 3;
  reshape(&b, s1, 3);
  print_tensor(&b);

  // broadcast add check
  struct Tensor c = arange(3);
  broadcast_sum(&b, &c);
  print_tensor(&b);
}
