rm -rf build || true
mkdir build || true
cd build
git clone -b v0.2.41 https://github.com/abetlen/llama-cpp-python
cd llama-cpp-python
git submodule update --init --recursive
# Build the wheel
pdm use 3.10 -i
CMAKE_ARGS="-DLLAMA_CUBLAS=on" pdm build -v