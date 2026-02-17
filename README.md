# Online Minimization of Polarization and Disagreement via Low-Rank Matrix Bandits

Official implementation of the ICLR 2026 paper:

**Online Minimization of Polarization and Disagreement via Low-Rank Matrix Bandits**  
Federico Cinus, Yuko Kuroki, Atsushi Miyauchi, Francesco Bonchi  
International Conference on Learning Representations (ICLR), 2026

---

## Installation

```bash
conda env create -f environment.yml
conda activate <env-name>
```

---

## Run Experiments

Run any of the provided scripts, for example:

```bash
python run-benchmark-ER.py
python run-benchmark-SB.py
python run-scalability-ER.py
python run-rsc-estimations.py
python run-sensitivity-REAL_GRAPHS.py
```

Figures are saved in the `figures/` directory.

---

## Citation

```bibtex
@inproceedings{cinus2026online,
  title={Online Minimization of Polarization and Disagreement via Low-Rank Matrix Bandits},
  author={Cinus, Federico and Kuroki, Yuko and Miyauchi, Atsushi and Bonchi, Francesco},
  booktitle={International Conference on Learning Representations},
  year={2026}
}
```
