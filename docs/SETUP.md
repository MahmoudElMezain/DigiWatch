# Environment setup

Start with one shared environment for CoralSCOP and the enhancement adapters that use the common PyTorch runtime. The [model registry](../Models/registry.json) records which of the 16 enhancement models are available. Use the [model guide](MODELS.md) for local file requirements and the optional runtime instructions below.

Each model's architecture code and local checkpoint folder are kept together under `Models/<Name>/`. Follow the [official enhancement repositories](../README.md#dehazing-models-and-repositories) and the [CoralSCOP repository](https://github.com/zhengziqiang/CoralSCOP) for their model releases and usage instructions.

## Requirements

- Windows or Linux, Git, and [Conda](https://docs.conda.io/projects/conda/en/stable/user-guide/install/index.html).
- Python 3.12, installed by the environment file below.
- An NVIDIA GPU is recommended, especially for CoralSCOP and diffusion models. CPU inference is available with `--device cpu`.
- A desktop session is useful for the manual selection preview. Automatic analysis and saved figures can run without a display.

The Windows reference runtime uses Python 3.12.9, PyTorch 2.6.0 with CUDA 12.4, torchvision 0.21.0, and the versions in [requirements.txt](../requirements.txt). Model availability and settings are documented in the registry and [model guide](MODELS.md). The Linux commands below use the matching official PyTorch wheels; Linux execution has not been separately validated.

## 1. Download the code

Run these commands in a Conda-enabled terminal. On Windows, use the Miniconda/Anaconda Prompt or a PowerShell session configured for Conda.

```console
git clone https://github.com/MahmoudElMezain/DigiWatch.git
cd DigiWatch
conda env create -f environments/environment.yml
conda activate coralwatch
```

The environment file installs Python, pip, and Tk for desktop previews. Install the compute-specific PyTorch build next, then the shared Python packages. This two-step setup keeps the same environment file usable on CPU and NVIDIA GPU systems. See [Conda's environment guide](https://docs.conda.io/projects/conda/en/stable/user-guide/tasks/manage-environments.html) for activation and environment management.

## 2. Install PyTorch

Choose **one** of these commands. Keep torch and torchvision at the matching versions shown. These commands follow the [official PyTorch 2.6 installation matrix](https://pytorch.org/get-started/previous-versions/#v260); torchaudio is not needed by this pipeline.

### NVIDIA GPU: Windows or Linux

Use the CUDA 12.4 build with a compatible NVIDIA driver:

```console
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124
```

If your GPU requires a different PyTorch/CUDA build, use the [official installation selector](https://pytorch.org/get-started/locally/). Changing this pair creates a different runtime from the validated configuration, so verify inference before collecting results.

### CPU: Windows or Linux

```console
python -m pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

CPU execution can take substantially longer. Selecting `--device cuda` with a CPU-only PyTorch build will fail; use `--device cpu` or install the GPU build.

## 3. Install the shared dependencies

```console
python -m pip install -r requirements.txt
python -m pip check
```

The shared packages cover image handling, plotting, CoralSCOP postprocessing, color matching, and the `einops`/`timm` utilities used by the added architectures. CLIP-UIE uses its trained diffusion generator without training-time learned prompts or a separate CLIP backbone. Optional runtime packages are described below; follow each author's repository for additional setup requirements.

## 4. Check the installation

These checks do not load model weights or process images:

```console
python -c "import sys, torch, torchvision, numpy, PIL, matplotlib, cv2, colour, tqdm, einops, timm; print('Python:', sys.version.split()[0]); print('PyTorch:', torch.__version__); print('torchvision:', torchvision.__version__); print('CUDA available:', torch.cuda.is_available())"
python "Full Pipeline.py" --help
```

For the GPU build, `CUDA available: True` confirms that PyTorch can use the GPU from this environment. For the CPU build, `False` is expected.

Before running inference, follow the authors' instructions in the [official repositories](../README.md#dehazing-models-and-repositories). See the [model file manifest](../Models/weights_manifest.json) for the required filenames. For example, the segmentation file belongs at `Models/CoralSCOP/weights/CoralSCOP.pth`, while MuLA-GAN uses `Models/MuLA-GAN/weights/MulaGAN.pth`. Architecture code sits beside these folders in each model's `architecture/` directory.

Use `--checkpoint` to override the CoralSCOP file or `--weights` to override the enhancement model's primary checkpoint. Models with multiple trained stages also require their companion files in the same checkpoint folder. For CCL-Net, both `best_net_G.pth` and `best_net_G_CC.pth` must be present.

For the MuLA-GAN example below, check that both required files are present:

```console
python tools/check_weights.py --model mulagan
```

Add `--sha256` to compare the files with the checkpoint versions tested for this pipeline. This check does not download or load the models. The checker resolves model-specific paths under `Models/`; `--all` checks the implemented models' recorded files. Use `--directory` only when the replacement directory has the same model subfolder structure.

## 5. Run an image

Start with MuLA-GAN and one automatically selected coral:

```console
python "Full Pipeline.py" "path/to/image.jpg" --model mulagan --mode coral-resize --analysis-mode automatic --coral-count 1
```

Replace `path/to/image.jpg` with your own image path and quote paths containing spaces. For a CPU installation, add `--device cpu`. The pipeline saves a new folder under `Output/` containing one figure for each analyzed coral.

Choose corals interactively from the ten highest-ranked CoralSCOP predictions:

```console
python "Full Pipeline.py" "path/to/image.jpg" --model mulagan --analysis-mode manual --top-k 10
```

Enter the displayed ranks separated by commas or spaces. Manual mode analyzes every selected rank; `--coral-count` controls automatic mode only. You can also provide ranks directly with `--selection 1 3 5`. Use `--no-show` on a machine without a graphical display, then open the printed candidate-preview path before entering the ranks.

DCD-UIE and CLIP-UIE each use **2,000 diffusion steps** by default; `--diffusion-steps` can change their sampling count. WF-Diff requires the authors' fixed **10 DDIM sampling steps** with its **2,000-step training noise schedule**; these are different quantities. Diffusion models can take much longer per image or patch. Keep sampling settings fixed when comparing results; the [model guide](MODELS.md) describes supported overrides.

## Optional SLURPP runtime

SLURPP runs both of its trained diffusion networks and a cross-latent decoder, with local Stable Diffusion 2 components. Install its additional packages in the shared environment or in a separate environment selected with `--model-python`:

```console
python -m pip install -r environments/requirements-slurpp.txt
python -m pip check
python tools/check_weights.py --model slurpp
```

Follow the [official SLURPP repository](https://github.com/kongdai123/SLURPP) for installation and the [model file manifest](../Models/weights_manifest.json) for its local file structure. Its required assets occupy about 9.43 GB. The default inference schedule uses one step; `--seed 2024` selects the author's reference seed.
