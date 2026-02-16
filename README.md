# Auditable Generation of S-Boxes
[![DOI](https://zenodo.org/badge/1148214113.svg)](https://doi.org/10.5281/zenodo.18662769)

> **Reproducibility Repository for:** *Auditable Generation of S-Boxes: Do Verifiable Initialization Constraints Affect Attainable Cryptographic Properties?*  
> Published in *Cryptologia* (2026)

This repository contains the essential reproducibility materials for the paper:

- **Generation scripts** (TRNG and PRNG initialization)
- **Optimization code** (Simulated Annealing algorithm)
- **20 optimized S-boxes** (10 TRNG + 10 PRNG)

---

## 📂 Repository Contents

```
SBox/
├── README.md                      # This file
├── generate_trng.py               # TRNG initialization (RANDOM.ORG)
├── generate_prng.py               # PRNG initialization (Mersenne Twister)
├── optimize_sbox.py               # Simulated annealing optimizer
│
└── sboxes/                        # 20 optimized S-boxes (JSON)
    ├── trng_sbox_001.json
    ├── trng_sbox_002.json
    ├── ...
    ├── trng_sbox_010.json
    ├── prng_sbox_001.json
    ├── prng_sbox_002.json
    ├── ...
    └── prng_sbox_010.json
```

---

## 🎯 Key Findings

Both initialization methods converge to comparable cryptographic properties:

| Property | TRNG (Verifiable) | PRNG (Conventional) |
|----------|-------------------|---------------------|
| Differential uniformity (δ) | 6.0 ± 0.0 | 6.0 ± 0.0 |
| Nonlinearity (NL) | 103.6 ± 1.2 | 103.8 ± 1.7 |
| Mean flipped output bits | 3.9996 ± 0.0349 | 4.0039 ± 0.0345 |
| Algebraic degree | 7 ± 0 | 7 ± 0 |

**Conclusion:** Verifiable initialization does not degrade S-box quality.

---

## 🚀 Usage

### Load an S-box

```python
import json

# Load a TRNG-initialized S-box
with open('sboxes/trng_sbox_001.json') as f:
    data = json.load(f)
    sbox = data['sbox']  # List of 256 integers [0-255]

# Access properties
print(f"Differential uniformity: {data['metrics']['differential_uniformity']}")
print(f"Nonlinearity: {data['metrics']['nonlinearity']}")
```

### Generate new S-boxes

```python
# Generate TRNG-initialized S-box
python generate_trng.py --output sboxes/new_trng.json

# Generate PRNG-initialized S-box  
python generate_prng.py --seed 12345 --output sboxes/new_prng.json

# Optimize an S-box
python optimize_sbox.py --input sboxes/new_trng.json --output sboxes/new_trng_optimized.json
```

---

## 📊 S-box File Format

Each JSON file contains:

```json
{
  "sbox": [147, 89, 234, ..., 201],     // 256 integers
  "initialization": {
    "type": "TRNG" | "PRNG",
    "source": "RANDOM.ORG" | "Mersenne Twister",
    ...
  },
  "metrics": {
    "differential_uniformity": 6,
    "nonlinearity": 104,
    "avalanche_mean": 4.002,
    "algebraic_degree": 7
  },
  "verification": {
    "sha256": "..."
  }
}
```

---

## 📄 Citation

```bibtex
@article{arboledas2026auditable,
  title={Auditable Generation of S-Boxes: Do Verifiable Initialization 
         Constraints Affect Attainable Cryptographic Properties?},
  author={Arboledas, David},
  journal={Cryptologia},
  year={2026},
  publisher={Taylor \& Francis},
  url={https://github.com/darboledas/SBox}
}
```

---

## 📧 Contact

**David Arboledas**  
Universidad de Alcalá  
david.arboledas@imagencientifica.es

---

## 📜 License

MIT License - See LICENSE file for details.

---

## ⚠️ Disclaimer

These S-boxes are for **research purposes only**. Not recommended for production use without extensive cryptanalysis.
