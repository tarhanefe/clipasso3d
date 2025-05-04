# From Pixels to Wireframes: 3D Reconstruction via CLIP-Based Sketch Abstraction

This repository contains the project proposal and pseudocode for “From Pixels to Wireframes,” a method for generating **3D sketch abstractions** using CLIP-based losses and Bézier curves on reconstructed surfaces.

## Table of Contents

- [Project Structure](#project-structure)  
- [Prerequisites](#prerequisites)  
- [Setup & Installation](#setup--installation)  
- [Building the PDF](#building-the-pdf)  
- [Usage](#usage)  
- [Contributing](#contributing)  
- [License](#license)  

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

## Prerequisites

- A modern **TeX distribution** (e.g., TeX Live 2023 or later, MiKTeX)  
- `latexmk` (for automated compilation)  
- `git` (to clone the repo)  

Optionally, you can build inside Docker or GitHub Codespaces if you prefer an isolated environment.

---

## Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/<your-username>/3d-sketch-abstraction.git
   cd 3d-sketch-abstraction
   ```

2. **Install TeX dependencies** (if not already installed):
   ```bash
   # On Ubuntu / Debian
   sudo apt update
   sudo apt install -y texlive-full latexmk
   ```

3. **(Optional) Docker-based build**  
   If you don’t want to install TeX locally, use our provided Dockerfile:
   ```bash
   docker build -t 3d-sketch-latex .
   docker run --rm -v "$PWD":/workspace -w /workspace 3d-sketch-latex      latexmk -pdf -interaction=nonstopmode main.tex
   ```

4. **(Optional) GitHub Codespaces**  
   Open this folder in GitHub Codespaces and it will automatically install TeX Live and `latexmk`.  

---

## Building the PDF

Once prerequisites are met, build your PDF with:

```bash
# Automatic rebuild on changes:
latexmk -pdf -interaction=nonstopmode   -file-line-error -halt-on-error main.tex
```

Or, for a one-shot compile:

```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

The final PDF will be generated as `main.pdf`.

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
