ARG CUDA_TAG=gpu
FROM mxnet/python:${CUDA_TAG}

ENV PYTHON_VERSION 2.7
ENV CUDA_MAJOR 9
ENV CUDA_MINOR 0
ENV CUDA_VERSION ${CUDA_MAJOR}-${CUDA_MINOR}
ENV CUDA_MXNET ${CUDA_MAJOR}${CUDA_MINOR}


RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libhdf5-serial-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python \
        python-dev \
        rsync \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python get-pip.py && \
rm get-pip.py

EXPOSE 6384

RUN pip install numpy==1.14.5

RUN pip install scikit-learn \
                scikit-image \
                scipy \
                nibabel \
                tqdm \
                SimpleITK \
                itk \
                opencv-python-headless \
                mxboard \
                cython \
                matplotlib \
                tensorflow

