# LL-DM

This project aims at creating a tool to generate and host a campaign for the game Dungeons and Dragons 5th edition.

## TODO
* [x] Create a basic UI
* [ ] Allow file upload
* [x] Set up document retrieval
* [ ] Set up chat summarization
* [ ] Set up audio generation (possible memory constraints)

## Installation
The following instructions are for **NVIDIA** cards.

We suggest using `conda` to simplify the dependency installation.

0. Install conda/miniconda/micromamba

1. Install the conda environment
```bash
conda env create -f environment.yml`
conda activate ll-dm
```

2. Build the optimized version of `llama-cpp-python`
```bash
./build-llama-cpp.sh`
```

3. Install the pip dependencies with pdm
```bash
pdm install
```

4. Download the LLM model
```bash
./download-model.sh
```

5. Start the UI
```bash
pdm run chainlit run app.py
```