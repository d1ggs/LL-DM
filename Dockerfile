FROM rocm/pytorch:rocm5.6_ubuntu20.04_py3.8_pytorch_2.0.1

ARG CMAKE_ARGS="-DLLAMA_CLBLAST=on"

WORKDIR /workspace

COPY environment.yml .

RUN conda env create -f environment.yml

RUN conda activate ll-dm && pdm install

COPY app.py .

CMD ["conda", "run", "--no-capture-output", "-n", "ll-dm", "chainlit", "run", "app.py"]