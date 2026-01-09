# FEMulator Pro User Manual

## 1. Overview

**FEMulator Pro** is distributed as a containerized application using **Docker**.
To simplify usage, all Docker operations are wrapped in **platform-specific helper scripts**.

> **Users never need to type Docker commands manually.**

There are **two sets of scripts**:

* **Windows users** → use `.bat` files
* **Linux users** → use `.sh` files

The scripts are **functionally equivalent**, differing only by operating system.

For users who prefer not to use Docker, a **manual installation** option is also available (see Section 10).

---

## 2. File Naming Convention (IMPORTANT)

Throughout this document:

* **Windows users** must use files ending in **`.bat`**
* **Linux users** must use files ending in **`.sh`**

| Purpose            | Windows           | Linux            |
| ------------------ | ----------------- | ---------------- |
| Start application  | `start.bat`       | `start.sh`       |
| Stop application   | `stop.bat`        | `stop.sh`        |
| Update application | `update.bat`      | `update.sh`      |
| Full rebuild       | `rebuild_all.bat` | `rebuild_all.sh` |
| Full removal       | `remove_all.bat`  | `remove_all.sh`  |

**Do not mix file types** (e.g. `.bat` files do not work on Linux).

---

## 3. Directory Structure (High Level)
```
project-root/
│
├── docker-compose-cpu.yml        # CPU-only container definition (all OS)
├── docker-compose-gpu.yml        # GPU-enabled container definition (all OS)
│
├── femulator.Dockerfile          # Main application image (all OS)
├── femulator.server.Dockerfile   # Base/server image (all OS)
│
├── requirements.txt              # Python dependencies (manual install)
│
├── start.bat                     # Windows start script
├── stop.bat                      # Windows stop script
├── update.bat                    # Windows update script
├── rebuild_all.bat               # Windows full rebuild
├── remove_all.bat                # Windows full cleanup
│
├── start.sh                      # Linux start script
├── stop.sh                       # Linux stop script
├── update.sh                     # Linux update script
├── rebuild_all.sh                # Linux full rebuild
├── remove_all.sh                 # Linux full cleanup
│
└── README.md
```

Dockerfiles and compose files are **shared across platforms**.
Only the scripts are OS-specific.

---

## 4. Prerequisites

### 4.1 Windows Users

* Windows 10 or newer
* **Docker Desktop** installed and running
* (Optional) NVIDIA GPU with WSL2 GPU support enabled

> **Windows users only use `.bat` files.**

---

### 4.2 Linux Users

* Linux system with Docker installed
* Docker Compose plugin available
* (Optional) NVIDIA GPU with `nvidia-container-toolkit`

> **Linux users only use `.sh` files.**

Before first use, Linux users must run once:
```bash
chmod +x *.sh
```

---

## 5. Core Concepts (No Docker Knowledge Required)

* **Image**: Packaged version of the application
* **Container**: A running instance of the application
* **CPU mode**: Works on all systems
* **GPU mode**: Used only when NVIDIA hardware is available

You do **not** need to understand Docker to use the scripts.

---

## 6. Script Reference (User-Level)

Each section below clearly states **which file Windows users run and which file Linux users run**.

---

### 6.1 Start the Application

**Purpose:**
Starts FEMulator Pro.

**Windows users:**
```bat
start.bat
```

**Linux users:**
```bash
./start.sh
```

**What it does:**

* Creates a Docker network (first run only)
* Starts the application container
* Uses CPU mode by default
* Uses GPU mode only when explicitly supported

**When to use:**

* Normal daily usage
* After a system reboot
* After stopping the application

---

### 6.2 Stop the Application

**Purpose:**
Stops the running container without deleting anything.

**Windows users:**
```bat
stop.bat
```

**Linux users:**
```bash
./stop.sh
```

**What it does:**

* Gracefully stops the application
* Keeps images intact
* Allows fast restart later

---

### 6.3 Update the Application (Soft Rebuild)

**Purpose:**
Rebuilds the main application image while keeping the base image.

**Windows users:**
```bat
update.bat
```

**Linux users:**
```bash
./update.sh
```

**What it does:**

* Stops the container
* Rebuilds the application image
* Uses cached layers when possible
* Faster than a full rebuild

**When to use:**

* Application code changes
* Minor updates

---

### 6.4 Full Clean Rebuild (Hard Reset)

**Purpose:**
Rebuilds **everything from scratch**.

**Windows users:**
```bat
rebuild_all.bat
```

**Linux users:**
```bash
./rebuild_all.sh
```

**What it does:**

* Stops containers
* Removes containers
* Removes all FEMulator images
* Rebuilds everything with no cache

**When to use:**

* Dependency changes

---

### 6.5 Remove Everything (Uninstall)

**Purpose:**
Completely removes FEMulator Pro from Docker.

**Windows users:**
```bat
remove_all.bat
```

**Linux users:**
```bash
./remove_all.sh
```

**What it does:**

* Stops containers
* Deletes containers
* Deletes images
* Leaves no FEMulator artifacts behind

---

## 7. CPU vs GPU Behavior

This behavior is identical on Windows and Linux.

### CPU Mode

* Always available
* Requires no special hardware
* Safe default on all systems

### GPU Mode

* Requires NVIDIA hardware
* Requires Docker GPU support
* Never forced on unsupported systems

Users do **not** need to choose manually.

---

## 8. Typical Usage Scenarios

### First-Time User

* **Windows:** `start.bat`
* **Linux:** `./start.sh`

---

### Daily Use

* Start → Use → Stop

---

### Update Application

* Stop → Update → Start

---

### Something Went Wrong

* Stop → Rebuild All → Start

---

### Remove Everything

* Run Remove All script for your OS

---

## 9. Summary

* **Windows users:** use `.bat` files only
* **Linux users:** use `.sh` files only
* Dockerfiles and compose files are shared
* Scripts provide a safe, reproducible workflow

This design allows **non-technical reviewers** to operate the system confidently on either platform.

---

## 10. Manual Installation (Without Docker)

For users who prefer to run FEMulator Pro directly on their system without Docker.

---

### 10.1 Prerequisites

#### All Systems

* Python 3.12 or newer
* pip package manager

#### For GPU Support (Optional)

* NVIDIA GPU with CUDA 13.x drivers installed
* CUDA Toolkit 13.x

#### Additional System Dependencies

* **WeasyPrint** requires system libraries: GTK, Pango, Cairo
* **Playwright** requires Chromium browser

---

### 10.2 Installation Steps

#### Step 1: Create a Virtual Environment

**Windows:**
```bat
python -m venv venv
venv\Scripts\activate
```

**Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

---

#### Step 2: Install Python Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

#### Step 3: Install Playwright Browser
```bash
playwright install chromium
```

---

#### Step 4: Install System Dependencies for WeasyPrint

**Windows:**

Install GTK3 runtime from https://github.com/nickvidal/gvsbuild/releases or use MSYS2.

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install libpango-1.0-0 libpangocairo-1.0-0 libgdk-pixbuf2.0-0 libffi-dev shared-mime-info
```

**Linux (Fedora/RHEL):**
```bash
sudo dnf install pango gdk-pixbuf2
```

---

### 10.3 Running the Application

After installation, start the server from the project root:
```bash
uvicorn main:app --host 0.0.0.0 --port 5867
```

Then open your browser to `http://localhost:5867`.

---

### 10.4 CPU-Only Installation

If you do not have an NVIDIA GPU, replace the GPU packages in `requirements.txt`:
```
# Remove these lines:
cupy-cuda13x
numba-cuda
cuda-python

# Add this line instead:
cupy
```

Or simply install without GPU packages and the application will fall back to CPU mode.

---

### 10.5 Troubleshooting Manual Installation

| Issue | Solution |
| ----- | -------- |
| `cupy` fails to install | Ensure CUDA Toolkit matches the `cupy-cuda13x` version |
| WeasyPrint errors | Install system GTK/Pango libraries |
| Playwright errors | Run `playwright install chromium` |
| Import errors | Verify virtual environment is activated |

---
