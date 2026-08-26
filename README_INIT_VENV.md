# Create virtual environment and install dependencies
### Create virtual environment and install base dependencies
```
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install torch
```
### Standard installation
(needs to be re-run after making changes to the code)
```
pip install .
```
### Optional - include testing dependencies (pytest, pytest-mock, requests)
```
pip install ".[test]"
```
### Basic installation to "editable" mode
(No need to re-run this command after making changes to the code, no re-installation needed.)
```
pip install -e .
```
### Optional - include testing dependencies (pytest, pytest-mock, requests)
```
pip install -e ".[test]"
```
### Test torch is available
python -c "import torch; print('CUDA:', torch.cuda.is_available()); print('MPS (Apple Silicon):', getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available())"

### What is Editable Mode (`pip install -e .`)?

**Editable mode** (short for `--editable`) installs a Python package directly from its source code directory by creating a link (or reference) in your virtual environment's `site-packages` directory pointing back to your project directory.

---

### Standard Install vs. Editable Install

| Feature | Standard Install (`pip install .`) | Editable Install (`pip install -e .`) |
| :--- | :--- | :--- |
| **File Location** | Copies files into `.venv/.../site-packages/` | Links directly to your local repo folder |
| **Code Edits** | Require running `pip install .` again after every code change | **Immediate effect** — edits in `nam/` take effect instantly |
| **Dependencies** | Installs dependencies listed in `pyproject.toml` | Installs dependencies listed in `pyproject.toml` |
| **CLI Commands** | Installs fixed entry point executables (`nam`, `nam-full`) | Installs entry point executables linked to your live code |
| **Best For** | Final deployment, production, or using 3rd-party packages | **Active package development & debugging** |

---

### Key Benefits

1. **No Re-installation Needed**: Whenever you edit a file inside [`nam/`](file:///Users/james/Code/neural-amp-modeler/nam), any test (`pytest`) or CLI command (`nam`) automatically uses your updated code.
2. **Dependency Resolution**: `pip` still reads [`pyproject.toml`](file:///Users/james/Code/neural-amp-modeler/pyproject.toml), resolves all dependencies (e.g. `torch`, `customtkinter`), and installs them into `.venv`.
3. **CLI Tools Linked to Source**: Entry point scripts configured in `pyproject.toml` (like `nam = "nam.cli:nam_gui"`) execute your local code directly.

---

### How It Works Under the Hood

Under standard Python packaging standards (PEP 660 / setuptools), `pip install -e .` creates a `.pth` file or dynamic import hook in `.venv/lib/python3.x/site-packages/`. When Python executes `import nam`, Python's module loader reads this link and loads the code directly from `/Users/james/Code/neural-amp-modeler/nam`.