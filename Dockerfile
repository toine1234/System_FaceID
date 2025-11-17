FROM python:3.10
WORKDIR /System_FaceID
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    python3-dev \
    libboost-all-dev \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*
COPY System_FaceID/requirements.txt ./
RUN pip install --upgrade pip
RUN pip install -r requirements.txt
COPY System_FaceID/. ./
CMD ["python", "app.py"]
