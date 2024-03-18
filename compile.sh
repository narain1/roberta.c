rm -r build
mkdir build && cd build
cmake .. -D CMAKE_BUILD_TYPE=Debug
cmake --build .
mv Roberta .. && cd ..
./Roberta
