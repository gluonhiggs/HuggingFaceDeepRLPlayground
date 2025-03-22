# HuggingFaceDeepRLPlayground
A playground for the the Hugging Face Deep RL Course if you have a decent local environment and want to avoid Google Colab Crashes.

## Pre-requisites
- Linux
- Python
- Pip
- Pyenv

## Installation
```bash
git clone
cd HuggingFaceDeepRLPlayground
sudo apt update
sudo apt install -y swig cmake python3-opengl ffmpeg xvfb nvidia-cuda-toolkit

```

```bash
pyenv virtualenv 3.11.5 DeepRLCourse-dev
pyenv activate DeepRLCourse-dev
python -m pip install -r requirements.txt
```

## Verify Installation
```bash
python test.py
```
The output should look like this:
```
True
0
NVIDIA GeForce RTX 3060 Laptop GPU
```

## Running the Course
Create a Jupyter Kernel for the virtual environment
```bash
python -m ipykernel install --user --name DeepRLCourse-dev --display-name "DeepRLCourse"
```

Start Jupyter
```bash
jupyter notebook
```

Select the Kernel `DeepRLCourse` and open the notebook.