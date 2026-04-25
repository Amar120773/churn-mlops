FROM python:3.10

WORKDIR /app

COPY . .

RUN pip install --upgrade pip && \
    pip install pandas scikit-learn mlflow streamlit

EXPOSE 8501

CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]