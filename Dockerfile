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
COPY svm_model.pkl /app/

COPY pfsense4-udp4-1195-config.ovpn /vpn_config/

RUN python3 -m venv venv

RUN /bin/bash -c "source venv/bin/activate && pip install --upgrade pip && pip install flask numpy pandas scikit-image dlib opencv-python-headless Pillow scikit-learn"

ENV VIRTUAL_ENV=/app/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

EXPOSE 5000

CMD ["python", "app.py"]
