# X-ray 2D-3D Registration & Vascular Analysis Toolkit

A small computer vision toolkit covering the core building blocks of
image-guided intervention pipelines: segmentation, 2D-3D registration,
tracking, and deployment optimization (ONNX export, quantization,
profiling, validation).

Built as a hands-on exploration of the techniques used in real-time
X-ray guided interventional imaging — segmentation of vascular
structures, registering a 3D pre-operative model onto live 2D
projections, tracking anatomy across a sequence, and preparing models
for real-time embedded inference.

## Motivation

This project was built to get hands-on experience with the specific
technical stack behind image-guided intervention systems: 2D-3D
registration between a pre-operative 3D volume and intra-operative 2D
projections, and the deployment constraints (latency, memory) that
come with running these models in real time.

## Modules

| Module | Status | Description |
|---|---|---|
| `src/registration/` | In progress | DRR generation + classical 2D-3D pose registration, built on [nanodrr](https://github.com/eigenvivek/nanodrr), using the public [DeepFluoro](https://github.com/rg2/DeepFluoroLabeling-IPCAI2020) benchmark |
| `src/segmentation/` | Planned | Coronary vessel segmentation on the [ARCADE](https://arcade.grand-challenge.org/) dataset |
| `src/tracking/` | Planned | Optical-flow-based tracking on synthetically deformed sequences |
| `src/deployment/` | Planned | ONNX export, quantization, latency/memory profiling, pre/post-quantization validation |

## Why nanodrr

[nanodrr](https://github.com/eigenvivek/nanodrr) is a pure-PyTorch,
auto-differentiable digitally reconstructed radiograph (DRR) renderer (a
faster reimplementation of the original DiffDRR, by the same author, with
no PyTorch3D dependency — installs with a plain `pip install`, no
compilation or CUDA toolkit needed). It lets us:

- Generate a simulated X-ray projection from a 3D CT volume at any pose
- Backpropagate through the rendering process itself, so a pose can be
  *optimized* directly by gradient descent (classical registration)
- Compare a "moving" DRR against a "fixed" target X-ray using a
  normalized cross-correlation loss, iterating until they align — the
  same forward-simulate-and-compare structure used in recent
  interventional-imaging registration research (e.g. DiffPose,
  Gopalakrishnan et al., CVPR 2024)

## Setup

```bash
pip install -r requirements.txt
```

No PyTorch3D, no conda-only step, no CUDA toolkit required — this installs
the same way on Windows, macOS, or Linux.

## Roadmap

- [x] Repo scaffold
- [ ] DRR generation demo (built-in example CT)
- [ ] Classical pose optimization (gradient descent through DiffDRR)
- [ ] Learned pose estimation (small CNN baseline)
- [ ] Vessel segmentation on ARCADE
- [ ] Synthetic-motion tracking demo
- [ ] ONNX export + quantization + profiling + validation
- [ ] Results write-up

## Limitations

This is a learning/portfolio project, not a clinical or validated
system. The registration module uses a generic example CT volume, not
patient-specific interventional data. Results are meant to demonstrate
understanding of the techniques, not clinical-grade performance.
