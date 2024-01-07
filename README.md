On AMD it is necessary to install `llama-cpp-python` with

```CMAKE_ARGS="-DLLAMA_CLBLAST=on" pip install "llama-cpp-python[server]" --upgrade --force-reinstall --no-cache-dir --verbose```

potentially after installing the conda env