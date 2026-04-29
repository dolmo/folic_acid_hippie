FROM pytorch/pytorch:2.3.1-cuda11.8-cudnn8-runtime

USER root

WORKDIR /src

# Install git (required to pip install from GitHub)
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install HIPPIE as a pip dependency
RUN pip install git+https://github.com/braingeneers/HIPPIE.git

# Install folic acid analysis dependencies
RUN pip install \
    numpy \
    pandas \
    scipy \
    matplotlib \
    seaborn \
    scikit-learn \
    umap-learn \
    hdbscan

# Copy the folic acid analysis code
COPY . /src/

# Set environment variables for paths
ENV HIPPIE_INPUT_DIR=/data/FA_T4
ENV HIPPIE_INPUT_T3_DIR=/data/FA_T3
ENV HIPPIE_RESULTS_DIR=/results/FA_T4
ENV HIPPIE_RESULTS_T3_DIR=/results/FA_T3

CMD ["/bin/bash"]