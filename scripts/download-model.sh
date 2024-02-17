# !/bin/bash

############################################
# This script is used to download the model
# and place it in the models directory.
############################################

# Move to the root directory of the project
cd ..

# Create models folder if it doesn't exist
mkdir -p models
cd models

# Download the quantized neural-chat-7B-v3-3 model in GGUF format to use with llama-cpp
wget https://huggingface.co/TheBloke/neural-chat-7B-v3-3-GGUF/resolve/main/neural-chat-7b-v3-3.Q5_K_M.gguf