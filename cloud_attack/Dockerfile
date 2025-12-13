FROM pytorch/pytorch:2.1.2-cuda11.8-cudnn8-runtime

WORKDIR /app

RUN pip install --no-cache-dir \
    fastapi==0.109.0 \
    uvicorn==0.27.0 \
    torchattacks==3.5.1 \
    google-generativeai>=0.5.0 \
    Pillow==10.2.0 \
    pandas==2.1.4 \
    numpy==1.26.0

COPY app/ ./app/

EXPOSE 8080
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8080"]