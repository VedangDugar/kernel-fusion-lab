# CPU-only development image.
#
# Triton kernels are executed here through the Triton interpreter
# (TRITON_INTERPRET=1), which runs kernel bodies on the CPU via NumPy instead of
# compiling to PTX. That validates kernel logic without a GPU, but produces no
# meaningful timing information. Wall-clock benchmarks are run separately on a
# real GPU via notebooks/colab_benchmark.ipynb.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /work

# torch is pulled from the CPU-only index so we do not download ~2GB of CUDA
# libraries that cannot be used in this container anyway.
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
 && pip install --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install -r /tmp/requirements.txt

COPY . /work

CMD ["bash"]
