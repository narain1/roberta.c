#include <stdio.h>

void updateValue(int *valueRef) {
    // Update the value pointed to by valueRef
    *valueRef = 10; // for example
}

int main() {
    int value = 5;
    
    printf("Before calling updateValue: %d\n", value);
    
    // Pass the address of 'value' to the function
    updateValue(&value);
    
    printf("After calling updateValue: %d\n", value);
    
    return 0;
}

