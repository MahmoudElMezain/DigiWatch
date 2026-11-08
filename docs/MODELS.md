# Model guide

The README contains one [table of all 16 official enhancement repositories](../README.md#dehazing-models-and-repositories). Follow each author's repository for model releases, installation instructions, and usage terms. Segmentation uses the separate [CoralSCOP repository](https://github.com/zhengziqiang/CoralSCOP).

## Model folders

Segmentation and enhancement models share one layout:

```text
Models/
├── registry.json
├── weights_manifest.json
├── CoralSCOP/
│   ├── architecture/
│   └── weights/
│       └── CoralSCOP.pth
├── MuLA-GAN/
│   ├── architecture/
│   └── weights/
│       └── MulaGAN.pth
└── CCL-Net/
    ├── architecture/
    └── weights/
        ├── best_net_G.pth
        └── best_net_G_CC.pth
```

Every model keeps its architecture and trained files together under its own directory. Required files are recorded in the [model file manifest](../Models/weights_manifest.json). Retained upstream license files accompany the architecture code.

The [registry](../Models/registry.json) records all registered enhancement names and their availability. An entry with `available: true` has an implemented adapter; its required local files must still be installed. Entries that remain unavailable give a reason. Model names are matched without case or punctuation differences, so `MuLA-GAN` and `mulagan` select the same model. Unknown or unavailable names return "not available in the current pipeline" and list the available choices.

## Setup and model selection

Use the [shared environment guide](SETUP.md) first. Adapters based on the common PyTorch runtime share that environment. Follow the authors' repositories for additional model-specific setup; use `--model-python` to select a separate environment when needed.

For example, analyze one coral using CCL-Net:

```console
python tools/check_weights.py --model ccl-net
python "Full Pipeline.py" "path/to/image.jpg" --model ccl-net --mode coral-resize --coral-count 1
```

The checkpoint checker includes CoralSCOP and every recorded stage of the selected enhancement model. It resolves relative paths under `Models/`. Add `--sha256` to compare the files against the tested reference versions. See the [model file manifest](../Models/weights_manifest.json) for filenames.

Use `--checkpoint` to override the CoralSCOP path or `--weights` to override the enhancement model's primary checkpoint. Additional stages are loaded from the primary checkpoint's folder using the filenames recorded in the registry. For CCL-Net, the haze-removal and color-correction files are both required.

## Diffusion settings

DCD-UIE and CLIP-UIE default to **2,000 sampling steps**. WF-Diff follows the authors' **10 DDIM steps** while retaining the **2,000-step training noise schedule**. A training schedule length and an inference sampling count are different settings.

`--diffusion-steps` changes the sampling count for adapters that support it, including DCD-UIE and CLIP-UIE. WF-Diff requires its fixed 10-step sampler. Sampling changes can affect both runtime and the generated image; keep settings fixed for comparisons. Each patch is a separate model input, so smaller overlap strides and larger patch grids can substantially increase diffusion processing time.

## Models with additional input requirements

**UVE-Net:** the official large video model runs on a one-frame sequence (`T=1`) in this pipeline. The adapter does not invent neighboring frames. Single-frame enhancement does not provide the temporal context used in a video benchmark.


Refer to the registry and official repositories for model settings and environment requirements.
