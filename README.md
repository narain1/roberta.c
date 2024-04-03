# RoBERTa Embedding

This project demonstrates how to perform embedding creation using a simplified RoBERTa model in pure C. 

### Overview

The code implements a portion of the forward pass of the RoBERTa model, focusing on the embedding stage. It uses a struct Buffer to hold intermediate tensor representations and performs operations such as mapping tokens to embeddings and adding positional embeddings to word embeddings.

### Prerequisites

A C compiler supporting C11 (e.g., GCC or Clang)
Basic knowledge of command-line operations

### Structure

    Tensor.h: Header file for tensor operations.
    RobertaModel.h: Header file containing the RoBERTa model structure.
    main.c: Contains the main function and demonstrates the usage of the forward function.
    Additional files for model implementation (not provided here but would include definitions for functions like map_embeddings and arange).

### Compilation Instructions

    Ensure Your Compiler is Installed: Verify that GCC or Clang is installed on your system. You can check this by running gcc --version or clang --version in your terminal.

    Compile the Source Code: Navigate to the source code directory in your terminal. Compile the code using the following command, replacing <your_compiler> with gcc or clang as appropriate:

```
<your_compiler> -std=c11 -mavx -o roberta_embedding main.c -lm
```

Run the Program: After successful compilation, you can run the program:

```
./roberta_embedding
```

### Usage Example

In main.c, you can set up a simple example to utilize the forward function:

```C
#include "RobertaModel.h"
#include "Tensor.h"
// Assume other necessary includes

int main() {
    // Example token array and its size
    int tokens[] = { /* Your tokenized input here */ };
    int n_tokens = sizeof(tokens) / sizeof(tokens[0]);

    // Initialize your model and buffer here
    // Assuming functions for initialization are defined

    struct Model *model = /* Initialize your model */;
    forward(model, tokens, n_tokens);

    // Clean-up code for your model and tensors
    return 0;
}
```

This usage example is hypothetical and assumes the existence of model initialization and clean-up code.
