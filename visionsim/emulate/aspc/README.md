# ASPC

## Setup
### 1. Create virtual environment (for Powershell)
```bash
python -m venv visionsim
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass  
.\visionsim\Scripts\Activate.ps1
```
### 2. Install requirements
```bash
pip install pint numpy matplotlib tqdm pyYAML scipy torch opencv-python ruamel.yaml
```

## Run commands

### 1. Test sources, hist, sensors
```bash
python .\visionsim\emulate\aspc\main_sources_hist.py
```

### 2. Formatting and linting
```bash
invoke lint
invoke format
```