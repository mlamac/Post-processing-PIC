# CLAUDE.md

## Overview

Standalone script for post-processing **EPOCH quasi-3D** LWFA simulations. Produces 2-panel HTML animation: density + laser field (top), density-colored phase space with momentum lineout (bottom).

## Usage

```bash
pip install -r requirements.txt
python post-process-lwfa.py  # Run in directory with SDF files
```

Requires `sdf_helper` from EPOCH (`make sdfutils` in epoch2d/).

## Key Physics

**Normalization order matters:** Normalize E_y, E_z individually before computing E_tot = sqrt(E_y² + E_z²).

**Quasi-3D modes:** `F(x,r,θ) = F₀(x,r) + F₁(x,r)cos(θ) + F₁ᵢ(x,r)sin(θ)`

**Units:** Length in λ₀, density in n_c, momentum in m_e c, transverse E-field in m_e c ω₀/e.

## SDF Field Names

- Fields: `Electric_Field_Modes_E{r,x}m_{real,imag}`
- Density: `Derived_Number_Density_Subset_total_e`
- Grid: `Grid_Grid_mid.data[0]` (x), `[1]` (r)
- Particles: `Grid_Particles_{species}`, `Particles_Px_{species}`, `Particles_Weight_{species}`

## Configuration

Edit `CONFIG` dict at top of script: `lambda0_um`, `a0`, `n_over_nc`, `species`, file patterns, visualization limits.
