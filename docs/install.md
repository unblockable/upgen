# Installation

This file documents the installation of UPGen project.
Please follow the following steps carefully.
In case of any problem or troubleshooting, create an issue on GitHub.

---

## Procedure

1. Clone the repo

```
[~/] $ git clone https://github.com/unblockable/upgen.git
```

2. Create a virtual environment and enter it (optional, recommended)

```
[~/] $ python -m venv venv
[~/] $ source venv/bin/activate 
```

3. Go to upgen repo and install all the required dependencies

```
[~/](venv) $ cd upgen
[~/upgen](venv) $ pip install -r requirements.txt
```

4. Verify installation

```
[~/upgen](venv) $ cd src
[~/upgen/src](venv) $ python generate.py --help
```

This completes the installation process of UPGen.

---
