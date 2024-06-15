FROM ubuntu:latest

RUN apt-get update && apt-get install -y \
    openssh-client \
    openvpn \
    python3 \
    python3-pip \
    python3-venv \
    python3-dev \
    cmake \
    build-essential \
    libboost-python-dev \
    libboost-thread-dev \
    libboost-system-dev \
    libboost-filesystem-dev \
    libboost-iostreams-dev \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    zlib1g-dev \
    libbz2-dev \
    libxml2-dev \
    libxslt-dev \
    libffi-dev \
    libssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY app.py /app/
COPY predict.py /app/
COPY svm_model_without_hog.pkl /app/
COPY svm_model_with_hog.pkl /app/
COPY dt_model_without_hog.pkl /app/
COPY dt_model_with_hog.pkl /app/
COPY rf_model_without_hog.pkl /app/
COPY rf_model_with_hog.pkl /app/
COPY gunicorn.conf.py /app/
COPY index.html /app/

RUN apt-get update && apt-get install -y curl 

RUN python3 -m venv venv
RUN /bin/bash -c "source venv/bin/activate && \
    pip install --upgrade pip && \
    pip install flask numpy pandas scikit-image dlib opencv-python-headless Pillow scikit-learn gunicorn"

EXPOSE 5000

CMD ["/bin/bash", "-c", "source venv/bin/activate && gunicorn --config gunicorn.conf.py app:app"]
