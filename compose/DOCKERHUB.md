# FEMulator Pro v1.0

A containerized deployment of the Femulator Pro application. This setup allows you to run the Femulator environment with all dependencies pre-configured.

---

## Quick Start

The easiest way to run Femulator is using **Docker Compose**. 

### 1. Prerequisites
* Install **Docker Desktop** (Windows/Mac) or **Docker Engine** (Linux). [Get Docker here](https://docs.docker.com/get-docker/).

### 2. Create the Configuration
Create a folder on your computer and save the following content into a file named `docker-compose.yml`:

```yaml
services:
  femulator:
    image: logus2k/femulator:1.0
    container_name: femulator
    hostname: femulator
    restart: unless-stopped
    logging:
      options:
        max-size: "10m"
        max-file: "3"
    ports:
      - "5868:5868"
    networks:
      - femulator_network

networks:
  femulator_network:
    driver: bridge

```

### 3. Launch the Application

Open your terminal or command prompt in that folder and run:

```bash
docker-compose up -d

```

The application will download the necessary images and start. You can now access the service via `localhost:5868`.

---

## 🛠 Management Commands

**View Application Logs** If you need to check the status or troubleshoot:

```bash
docker logs -f femulator

```

**Stop the Application** To stop and remove the containers (your data inside the container may be lost if not volume-mapped):

```bash
docker-compose down

```

**Update to the Latest Version** To ensure you are running the most recent version of the image:

```bash
docker-compose pull
docker-compose up -d

```

---

## Technical Details

| Detail | Setting |
| --- | --- |
| **Image** | `logus2k/femulator:1.0` |
| **Default Port** | `5868` |
| **Restart Policy** | `unless-stopped` |

---
