# Берём официальный образ Airflow 2.9
FROM apache/airflow:2.9.2

# root нужен чтобы создать папку results
USER root
RUN mkdir -p /opt/airflow/results && \
    chown -R airflow:root /opt/airflow/results

USER airflow

# Копируем зависимости и устанавливаем
COPY requirements.txt /requirements.txt
RUN pip install --no-cache-dir -r /requirements.txt