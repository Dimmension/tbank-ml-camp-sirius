FROM nvcr.io/nvidia/tritonserver:23.07-py3
COPY triton_repository /triton_repository
WORKDIR /triton_repository
