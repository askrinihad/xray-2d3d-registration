# X-ray 2D-3D Registration & Vascular Analysis Toolkit

A computer vision toolkit for image-guided interventional imaging,
covering 2D-3D registration, learned pose estimation, vessel
segmentation, motion tracking, and deployment optimization for
real-time inference.

## Overview

The toolkit addresses the core technical problems in X-ray guided
interventional procedures: aligning a pre-operative 3D volume (CT) with
live intra-operative 2D projections (fluoroscopy), segmenting vascular
structures, tracking anatomy across a sequence under physiological
motion, and preparing models for real-time embedded inference. Two
approaches to registration are implemented and compared — classical
gradient-based optimization and a learned CNN — and the learned model
is optimized for deployment via ONNX export and quantization.

| Module | Status | Description |
|---|---|---|
| `src/registration/` | Complete, validated | DRR generation, classical gradient-based 2D-3D registration, and a learned CNN alternative with a held-out generalization test |
| `src/segmentation/` | Complete, validated | Binary coronary vessel segmentation (small U-Net) on the ARCADE dataset |
| `src/tracking/` | Complete, validated | Keypoint tracking and dense motion estimation on a synthetically deformed sequence, validated against known ground truth |
| `src/deployment/` | Complete, validated | ONNX export, INT8 quantization, latency/size profiling, and accuracy validation |

## Results summary

**Registration — classical vs. learned** (DeepFluoro, 5 training subjects,
1 held out entirely):

| | Rotation error | Translation error | Latency vs. classical |
|---|---|---|---|
| Learned CNN — seen subject | 6.6° | 13.1 mm | ~300x faster |
| Learned CNN — held-out subject | 10.3° | 26.4 mm | ~330x faster |

The classical method converges reliably (typically under 100 iterations)
but requires iterative optimization per pose. The learned model predicts
a pose in a single forward pass. The accuracy gap between seen and
held-out anatomy reflects the limited subject diversity available in
DeepFluoro (6 subjects total) rather than a flaw in the approach; for
reference, published state-of-the-art on this same dataset reaches
sub-millimeter accuracy, reflecting the difference in scale between a
research system and this project.

**Segmentation** (ARCADE, binary vessel mask, small U-Net, 15 epochs):

Validation Dice score of **0.75** (two independent runs: 0.7613, 0.7509).
Qualitative review shows accurate localization of main vessel trunks,
with fragmentation on thinner secondary branches and lower-contrast
frames — a specific, addressable limitation rather than a general
failure mode.

**Tracking and motion estimation** (synthetic sequence derived from a
real ARCADE frame, with known ground-truth deformation combining
respiratory- and cardiac-like motion components):

- 60/60 keypoints tracked successfully across all 20 frames with Lucas-Kanade
  optical flow, with no drift onto incorrect structures
- Dense motion estimation (Farneback optical flow) achieved a mean
  End-Point Error of **2.82 px** against the known ground truth, with no
  degrading trend across the sequence (ruling out accumulated drift)

**Deployment — PyTorch vs. ONNX vs. quantized ONNX**:

| Variant | Latency | Size | Rotation error | Translation error |
|---|---|---|---|---|
| PyTorch (FP32) | 1.31 ms | 275.8 KB | 6.58° | 13.12 mm |
| ONNX (FP32) | 0.89 ms | 274.0 KB | 6.58° | 13.12 mm |
| ONNX (INT8, quantized) | 0.66 ms | 80.7 KB | 6.44° | 12.97 mm |

ONNX export alone gives a 1.5x speedup with no accuracy change. INT8
quantization adds a further 1.3x speedup and a 70.6% size reduction,
with no measurable accuracy cost on held-out test poses.

## Setup

```bash
pip install -r requirements.txt
```

No PyTorch3D, no conda-only step, no CUDA toolkit required — installs the
same way on Windows, macOS, or Linux.

## Registration module

Two approaches to 2D-3D registration are implemented on the same task,
built on [nanodrr](https://github.com/eigenvivek/nanodrr) (a pure-PyTorch,
auto-differentiable DRR renderer) and the public
[DeepFluoro](https://github.com/rg2/DeepFluoroLabeling-IPCAI2020)
benchmark (Grupp et al., 2020).

**Classical** (`pose_estimation.py`): gradient-based optimization through
the differentiable renderer. A rendered DRR is compared against a target
image using normalized cross-correlation, and the pose is iteratively
updated by backpropagating the image similarity loss — the same
forward-simulate-and-compare structure used in recent interventional
registration research (e.g. DiffPose, Gopalakrishnan et al., CVPR 2024).

**Learned** (`train_pose_cnn.py`, `compare_classical_vs_learned.py`): a
small CNN trained to predict pose directly from a single image, in one
forward pass, using synthetic (image, pose) pairs generated on the fly.
Trained across 5 of DeepFluoro's 6 subjects, with the 6th held out
entirely to measure generalization to unseen anatomy rather than assume
it.

```bash
python src/registration/generate_drr.py
python src/registration/pose_estimation.py
python src/registration/train_pose_cnn.py              # or use notebooks/pose_cnn_colab.ipynb
python src/registration/compare_classical_vs_learned.py
```

## Segmentation module

Binary coronary vessel segmentation on
[ARCADE](https://zenodo.org/records/10390295) (CC0 license, Popov et al.,
published in *Scientific Data*): 1,200+ expert-labeled X-ray coronary
angiography images, originally annotated across 26 SYNTAX vessel-segment
categories. All 26 categories are collapsed into a single binary
vessel-vs-background mask for this version; full 26-class segmentation is
a natural extension.

```bash
python src/segmentation/download_arcade.py   # ~450 MB, CC0, no registration required
python src/segmentation/train.py             # small U-Net, ~15 epochs
python src/segmentation/evaluate.py
```

## Tracking module

Since real fluoroscopic video with known ground-truth motion doesn't
exist publicly, a synthetic sequence is generated from a real ARCADE
frame by applying a smooth, non-rigid deformation combining a slow
respiratory-like component and a faster cardiac-like component — giving
a known, exact ground truth to validate against.

**Tracking** (`track_and_estimate_motion.py`): sparse keypoints on
vessel structure followed across the sequence with Lucas-Kanade optical
flow. **Motion estimation**: dense per-pixel displacement between
consecutive frames with Farneback optical flow, evaluated against the
known deformation using End-Point Error (EPE), the standard metric in
the optical flow literature.

```bash
python src/tracking/generate_synthetic_sequence.py
python src/tracking/track_and_estimate_motion.py
```

## Deployment module

Prepares the learned registration CNN — the model relevant to real-time
use, unlike the classical iterative method — for embedded inference.

```bash
python src/deployment/export_onnx.py      # PyTorch -> ONNX
python src/deployment/quantize_onnx.py    # ONNX FP32 -> ONNX INT8 (dynamic quantization)
python src/deployment/profile_models.py   # measured latency and size
python src/deployment/validate_onnx.py    # accuracy check across all three variants
```

## Limitations

This is a portfolio project, not a validated clinical system. Results
are intended to demonstrate understanding of the underlying techniques
— 2D-3D registration, classical/learned comparison, segmentation,
motion tracking, and deployment optimization — rather than
clinical-grade performance.

- The registration generalization gap reflects the small number of
  subjects available in the public DeepFluoro dataset (6 total); a
  production model would require training data spanning substantially
  more subjects.
- Segmentation uses a public benchmark (ARCADE) rather than data from
  the specific target application (contrast-agent-free cardiovascular
  fluoroscopy), and collapses 26 clinically distinct vessel segments
  into a single binary class.
- Tracking and motion estimation are validated on a synthetic sequence
  with known ground truth rather than real fluoroscopic video, since
  paired video with known motion doesn't exist publicly; real video
  would introduce additional challenges (contrast variation, instrument
  occlusion) not present here.
