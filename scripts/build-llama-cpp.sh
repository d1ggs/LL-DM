#!/bin/bash

############################################
# This script is used to build the llama-cpp
# package and generate a wheel file.
############################################

# Move to the root directory of the project
cd ..

# Define the package name and version to build
PACKAGE_NAME="llama-cpp-python"
PACKAGE_VERSION="0.2.39" # You can adjust this as needed
WHEELS_DIR="./wheels"   # Directory to store the generated wheel

# Create a directory for the wheel if it doesn't exist
mkdir -p "${WHEELS_DIR}"

# Set the CMake arguments
export CMAKE_ARGS="-DLLAMA_CUBLAS=on"

# Build the wheel
pip wheel "${PACKAGE_NAME}==${PACKAGE_VERSION}" --no-deps --wheel-dir "${WHEELS_DIR}"

# Find the generated wheel file
# Assuming only one wheel is generated in the process for simplicity
PACKAGE_NAME=$(echo "$PACKAGE_NAME" | sed 's/-/_/g')
WHEEL_FILE=$(find "${WHEELS_DIR}" -name "${PACKAGE_NAME}-${PACKAGE_VERSION}-*.whl" | head -n 1)

if [ -z "${WHEEL_FILE}" ]; then
    echo "Wheel file not found."
    exit 1
else
    echo "Wheel file generated at: ${WHEEL_FILE}"

    # Remove the existing entry in the TOML file if it exists

    # Specify the TOML file path
    FILE_PATH="pyproject.toml"

    # Backup the original file before modification
    cp "$FILE_PATH" "${FILE_PATH}.bak"

    # Remove lines containing "llama-cpp-python" entries from the dependencies array
    sed -i '/llama-cpp-python/d' "$FILE_PATH"

    echo "Entries containing 'llama-cpp-python' have been removed from $FILE_PATH."

    pdm add $WHEEL_FILE
    # Optionally, add commands here to copy the wheel file to another location or perform additional actions
fi
