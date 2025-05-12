# From Pixels to Wireframes: 3D Reconstruction via CLIP-Based Sketch Abstraction

<p align="center">
  <img src="assets/pipeline.png" width=700px />
</p>

This repository contains the project proposal and pseudocode for “From Pixels to Wireframes,” a method for generating **3D sketch abstractions** using CLIP-based losses and Bézier curves on reconstructed surfaces.

---

# Project Structure

```
clipasso3d/
├── CLIP_/                    
├── data/
├── notebooks/
├── source/
├── .gitignore
├── .gitmodules
├── README.md
└── requirements.txt
```

---

# Setup & Installation

**Clone the repository:**
   ```bash
   git clone --recurse--submodules https://github.com/tarhanefe/clipasso3d.git
   cd clipasso3d
   ```

**Create a new Conda environment with python version 3.10:**
   ```bash
   conda create -n 3dsketch python=3.10 -y
   conda activate 3dsketch
   ```
**Install Pytorch for Cuda version 12.1:**
   ```bash
   pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu121
   ```

**Install library requirements:**
   ```bash
   pip install -r requirements.txt
   ```
---

# Demo

## Single View 3D Generations

### A snake
<p align="center">
  <img src="assets/snake.gif" width=300px />
</p>


<p align="center">
  <img src="data/snake.jpg" width=300px />
</p>

### A giraffe
<p align="center">
  <img src="assets/giraffe.gif" width=300px />
</p>


<p align="center">
  <img src="data/giraffe.jpg" width=300px />
</p>

### A dolphin
<p align="center">
  <img src="assets/dolphin.gif" width=300px />
</p>


<p align="center">
  <img src="data/dolphin.png" width=300px />
</p>

## Multi View 3D Generations

### The plant 
<p align="center">
  <img src="assets/tree.gif" width=500px />
</p>


<p align="center">
  <img src="assets/tree.png" width=300px />
</p>

<p align="center">
  <img src="assets/tree_gt.png" width=300px />
</p>
