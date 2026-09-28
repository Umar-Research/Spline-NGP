# A Tensor-Product B-Spline Framework for Instant Neural Graphics Primitives

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

This repository contains the official implementation of the paper:

> **A Tensor-Product B-Spline Framework for Instant Neural Graphics Primitives**  
> Umar Anjum, Shahid Saeed Siddiqi, Muhammad Zia Afzal  
> *Preprint / Under Review*

---

## 📌 Overview

Instant Neural Graphics Primitives (Instant-NGP) accelerated neural radiance field training through multiresolution hash-grid representations. While typically treated as an engineering heuristic for memory locality and fast hardware evaluation, this work bridges multiresolution grid representations with classical continuous approximation theory:

1. **Theoretical Equivalence Proof**: We prove that trilinear interpolation over structured grid cells is mathematically identical to an expansion in a degree-1 ($p=1$) tensor-product B-spline basis.
2. **$C^1$ Continuity Generalization**: We extend coordinate encodings to arbitrary degree $p \ge 1$ splines, deriving an explicit quadratic ($p=2$) tensor-product B-spline formulation using 27-point local stencils to provide continuous first derivatives across voxel boundaries.
3. **Unified Comparative Study**: Under identical ray-sampling schedules and volumetric rendering quadrature, our degree-1 spline identically replicates Instant-NGP, while the quadratic B-spline provides smoother optimization trajectories and reduces high-frequency artifacts during early iterations.

---

## 🔬 Mathematical Formulation

### 1. Degree-1 Equivalence (Instant-NGP)
For a local normalized coordinate $\tau \in [0, 1)^3$ within grid cell $i$, standard trilinear interpolation computes:
$$f_l(\mathbf{x}) = \sum_{\delta \in \{0, 1\}^3} w_\delta(\tau^{(l)}) \theta_{l, h_l(i^{(l)} + \delta)}$$


Because the cardinal degree-1 B-spline hat function is defined as $B_1(\xi) = \max(1 - \vert{}\xi\vert{}, 0)$, we establish:
$$\tau^\delta (1 - \tau)^{1 - \delta} \equiv B_1(\tau - \delta)$$

which proves exact equivalence to a degree-1 tensor-product B-spline field.

### 2. Quadratic ($p=2$) $C^1$-Continuous Extension
The quadratic cardinal B-spline kernel evaluated over interval coordinates $t \in [0, 1)$ utilizes three piecewise polynomial segments:
$$B_0(t) = \frac{(1 - t)^2}{2}, \quad B_1(t) = \frac{-2t^2 + 2t + 1}{2}, \quad B_2(t) = \frac{t^2}{2}$$


Evaluating over a symmetric 27-point neighborhood yields:
$$f_l(\mathbf{x}) = \sum_{a=0}^2 \sum_{b=0}^2 \sum_{c=0}^2 B_a(\tau_1^{(l)}) B_b(\tau_2^{(l)}) B_c(\tau_3^{(l)}) \, \theta_{l, h_l(i_1^{(l)} + a - 1, \, i_2^{(l)} + b - 1, \, i_3^{(l)} + c - 1)}$$

---

## 🚀 Getting Started

### Prerequisites
* Linux or Windows with CUDA support
* Python $\ge$ 3.10
* NVIDIA GPU with Compute Capability $\ge$ 7.5

### Installation

Clone the repository and install dependencies:
```bash
git clone [https://github.com/Umar-Research/Spline-NGP.git](https://github.com/Umar-Research/Spline-NGP.git)
cd spline-nerf
pip install -r requirements.txt