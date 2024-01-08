FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04
# replace CUDA version to your CUDA version.
# You can check your CUDA version with below.
# nvcc -V

RUN apt-get update
RUN apt-get install -y python3
RUN apt-get -y install python3-pip
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
# replace CUDA version to your CUDA version.

RUN mkdir bud-nougat
WORKDIR /bud-nougat

RUN pip3 install fastapi uvicorn[standard] fsspec[http]==2023.5.0 python-multipart
COPY . .
WORKDIR /bud-nougat

RUN python3 setup.py install

EXPOSE 8503

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8503"]
# Run this using 'docker run -it -d -p <YOUR PORT>:8503 --gpus all <IMAGE NAME>
