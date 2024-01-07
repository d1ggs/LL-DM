# LL-DM

This project aims at creating a tool to generate and host a campaign for the game Dungeons and Dragons 5th edition.

## TODO
[x] Create a basic UI
[ ] Allow file upload
[ ] Set up document retrieval
[ ] Set up chat summarization
[ ] Set up audio generation (possible memory constraints)

## Installation
The following instructions are for AMD cards. I will provide an NVIDIA version soon.

1. Make Sure you have the model files in the `models` folder
```bash
chmod +x download_models.sh
./download_models.sh
```
2. Install the python dependencies with conda
```bash
export CMAKE_ARGS="-DLLAMA_CLBLAST=on"
conda env create -f environment.yml
```
3. [Optional] If upgrading the environment, make sure to remove `llama-cpp-python` first, it needs to be built from source