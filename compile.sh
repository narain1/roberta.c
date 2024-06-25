# rm -r build
# mkdir build && cd build
# cmake .. -D CMAKE_BUILD_TYPE=Debug
# cmake --build .
# mv Roberta .. && cd ..
# ./Roberta
gcc -o Roberta Tensor.c roberta.c -march=native -fopenmp -lm -g
./Roberta
