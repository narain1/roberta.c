#include "Tensor.h"
#include <stdio.h>
#include <stdbool.h>
#include <stdlib.h>

#define print_bool(x) return "%s\n", (x) ? "true" : "false"

bool broadcast_check(struct Tensor *a, struct Tensor *b) {
  printf("%d, %d\n", a->ndim, b->ndim);

  if (b->ndim > a->ndim) {
    printf("number of tensor dimension mismatch %d %d\n", b->ndim, a->ndim);
    fprintf(stderr, "tensors not broadcastable %s %s\n", get_tensor_shape_str(a), get_tensor_shape_str(b));
    return false;
    // exit(EXIT_FAILURE);
  }
  int dim = 1, ld, rd;
  while (dim <= b->ndim) {
    ld = a->shape[a->ndim - dim];
    rd = b->shape[b->ndim - dim];
    if (!(rd != 1 || ld != rd)) {
      return false;
    }
    ++dim;
  }
  return true;
}

int main() {
  unsigned int s[2] = {5, 15};
  struct Tensor a = rand_tensor(s, 2);
  unsigned int sa[1] = {1};
  struct Tensor a1 = rand_tensor(sa, 1);

  printf("broadcast check: %s\n\n", broadcast_check(&a, &a1) ? "true" : "false");

  unsigned int s2[2] = {5, 4};
  struct Tensor b = rand_tensor(s2, 2);

  unsigned int s3[1] = {4};
  struct Tensor b1 = rand_tensor(s3, 1);

  printf("broadcast check: %s\n\n", broadcast_check(&b, &b1) ? "true" : "false");

  unsigned int s4[2] = {5, 4};
  struct Tensor c = rand_tensor(s4, 2);

  unsigned int s5[2] = {3, 4};
  struct Tensor c1 = rand_tensor(s5, 2);

  printf("broadcast check: %s\n\n", broadcast_check(&c, &c1) ? "true" : "false");

  unsigned int s6[2] = {5, 4};
  struct Tensor d = rand_tensor(s4, 2);

  unsigned int s7[3] = {3, 4, 1};
  struct Tensor d1 = rand_tensor(s5, 3);

  printf("broadcast check: %s\n\n", broadcast_check(&d, &d1) ? "true" : "false");

  unsigned int s8[3] = {15, 3, 5};
  struct Tensor d2 = rand_tensor(s4, 3);

  unsigned int s9[3] = {3, 5};
  struct Tensor d3 = rand_tensor(s5, 2);

  printf("broadcast check: %s\n\n", broadcast_check(&d2, &d3) ? "true" : "false");

  unsigned int s10[3] = {15, 3, 5};
  struct Tensor d4 = rand_tensor(s4, 3);

  unsigned int s11[3] = {15, 1, 5};
  struct Tensor d5 = rand_tensor(s5, 3);

  printf("broadcast check: %s\n\n", broadcast_check(&d4, &d5) ? "true" : "false");

  unsigned int s12[3] = {15, 3, 5};
  struct Tensor d6 = rand_tensor(s4, 3);

  unsigned int s13[3] = {3, 1};
  struct Tensor d7 = rand_tensor(s5, 1);

  printf("broadcast check: %s\n\n", broadcast_check(&d6, &d7) ? "true" : "false");


}
