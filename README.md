# From Pixels to Wireframes: 3D Reconstruction via CLIP-Based Sketch Abstraction

This repository contains the project proposal and pseudocode for “From Pixels to Wireframes,” a method for generating **3D sketch abstractions** using CLIP-based losses and Bézier curves on reconstructed surfaces.

## Table of Contents

- [Project Structure](#project-structure)  
- [Prerequisites](#prerequisites)  
- [Setup & Installation](#setup--installation)  
---

## Project Structure

```
.
├── pipeline.png              # Pipeline illustration
├── literature.bib            # Bibliography file
├── main.tex                  # LaTeX source (proposal)
├── appendices.tex            # Appendix with pseudocode
├── figures/
│   └── ...                   # Any additional figures
└── README.md                 # ← You are here
```

- **main.tex** — the core proposal document (title, abstract, introduction, method, related work, discussion).  
- **appendices.tex** — contains the pseudocode (App. A).  
- **literature.bib** — bibliography entries for all cited works.  
- **pipeline.png** — high-level overview of the pipeline.  


---

## Setup & Installation

**Clone the repository:**
   ```bash
   git clone https://github.com/tarhanefe/clipasso3d.git
   cd clipasso3d
   ```

**Create a new Conda environment with python version 3.10:**
   ```bash
   conda create -n 3dsketch python=3.10 -y
   conda activate 3dsketch
   ```
**Install Pytorch for Cuda version 12.1:**
   ```bash
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
   ```

**Install library requirements:**
   ```bash
   pip install -r requirements.txt
   ```

---

## Usage

- **Read the PDF**  
  ```bash
  xdg-open main.pdf   # Linux
  open main.pdf       # macOS
  start main.pdf      # Windows
  ```

- **Edit content**  
  - Modify sections in `main.tex`.  
  - Update pseudocode in `appendices.tex`.  
  - Add new figures under `figures/` and include them with `\includegraphics`.  

- **Add / Update references**  
  - Edit `literature.bib`  
  - Run `bibtex` or let `latexmk` handle it automatically.

---

## Contributing

Feel free to open issues or pull requests to:

- Refine the write-up  
- Update pseudocode or figures  
- Improve formatting or styling  

Please follow standard IEEE Tran `.bib` formatting for references.

---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.
