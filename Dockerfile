FROM pytorch/pytorch:2.4.1-cuda11.8-cudnn9-runtime

WORKDIR /app

COPY requirements_docker.txt .

RUN pip install --progress-bar=on -r requirements_docker.txt

COPY . .

CMD ["python", "api.py"]