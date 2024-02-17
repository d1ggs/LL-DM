# rm -rf build || true
# mkdir build || true
# cd build
# git clone -b v0.2.41 https://github.com/abetlen/llama-cpp-python
# cd llama-cpp-python
# git submodule update --init --recursive
# # Build the wheel
# pdm use 3.10 -i
# CMAKE_ARGS="-DLLAMA_CUBLAS=on" pdm build -v

#!/bin/bash

# Define the package name and version to build
PACKAGE_NAME="llama-cpp-python"
PACKAGE_VERSION="0.2.41" # You can adjust this as needed
WHEELS_DIR="./wheels"   # Directory to store the generated wheel

# Create a directory for the wheel if it doesn't exist
mkdir -p "${WHEELS_DIR}"

# Set the CMake arguments
export CMAKE_ARGS="-DLLAMA_CUBLAS=on"

# Build the wheel
pip wheel "${PACKAGE_NAME}==${PACKAGE_VERSION}" --no-deps --wheel-dir "${WHEELS_DIR}"

# Find the generated wheel file
# Assuming only one wheel is generated in the process for simplicity
WHEEL_FILE=$(find "${WHEELS_DIR}" -name "${PACKAGE_NAME}-${PACKAGE_VERSION}-*.whl" | head -n 1)

if [ -z "${WHEEL_FILE}" ]; then
    echo "Wheel file not found."
    exit 1
else
    echo "Wheel file generated at: ${WHEEL_FILE}"
    # Optionally, add commands here to copy the wheel file to another location or perform additional actions
fi
