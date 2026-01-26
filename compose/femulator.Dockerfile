FROM logus2k/femulator.server:latest

USER root

WORKDIR /src
COPY src/ ./

WORKDIR /src/app/server

EXPOSE 5868

CMD ["uvicorn", "fem_api_server:socket_app", "--host", "0.0.0.0", "--port", "5868"]
