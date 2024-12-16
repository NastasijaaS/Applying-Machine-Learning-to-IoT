FROM bitnami/spark:latest

COPY . /app
RUN cd /app && pip install -r requirements.txt

ENV SPARK_APPLICATION_PYTHON_LOCATION /app/streaming.py
ENV SPARK_MASTER_NAME spark
ENV SPARK_MASTER_PORT 7077
ENV SPARK_SUBMIT_ARGS "--packages org.apache.spark:spark-sql-kafka-0-10_2.12:3.5.0"

COPY submit.sh /
CMD ["/bin/bash", "/submit.sh"]
