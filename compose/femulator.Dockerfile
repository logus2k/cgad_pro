FROM femulator.server:1.0

USER root

WORKDIR /src
COPY src/ ./

WORKDIR /src/app/server

EXPOSE 5867

CMD ["uvicorn", "fem_api_server:socket_app", "--host", "0.0.0.0", "--port", "5867"]
