FROM continuumio/miniconda3:latest

WORKDIR /app

COPY conda-lock.yml .
RUN conda install -c conda-forge conda-lock -y && \
    conda-lock install -n mlops-kickoff-2 conda-lock.yml && \
    conda clean -afy

ENV PATH=/opt/conda/envs/mlops-kickoff-2/bin:$PATH

COPY . .

EXPOSE 8080

CMD ["uvicorn", "src.api:app", "--host", "0.0.0.0", "--port", "8080"]