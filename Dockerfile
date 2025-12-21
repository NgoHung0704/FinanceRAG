# Use conda base image which has better package management
FROM continuumio/miniconda3:latest

WORKDIR /workspace

# Copy requirements
COPY requirements.txt .

# Create conda environment
RUN conda create -n financerag python=3.11 -y

# Install dependencies using conda (includes build tools)
RUN conda install -n financerag -c conda-forge gcc_linux-64 -y && \
    /opt/conda/envs/financerag/bin/pip install -r requirements.txt && \
    /opt/conda/envs/financerag/bin/pip install pytrec_eval datasets jupyter jupyterlab

# Copy project
COPY . .

# Set Python path
ENV PYTHONPATH=/workspace
ENV PATH=/opt/conda/envs/financerag/bin:$PATH

EXPOSE 8888

CMD ["jupyter", "lab", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''"]
