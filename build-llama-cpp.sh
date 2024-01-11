git clone -b v0.2.28 https://github.com/abetlen/llama-cpp-python
cd llama-cpp-python
git submodule update --init --recursive
# Build the wheel
pdm use 3.10
CMAKE_ARGS="-DLLAMA_CLBLAST=on" pdm build -v