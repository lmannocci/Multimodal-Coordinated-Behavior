# Multimodal-Coordinated-Behavior

This repository contains the full pipeline and configuration files used to run the analysis on two separate datasets: **IO Russia** and **UK**.
Each dataset has its own configuration file and execution entry point, allowing the user to run the workflow independently for each case.

---

## Repository Structure

```
.
├── input_config_IORussia.py   # Configuration parameters for the IO Russia dataset
├── input_config_uk.py         # Configuration parameters for the UK dataset
├── main_IORussia.py           # Main script to run the pipeline on the IO Russia dataset
├── main_uk.py                 # Main script to run the pipeline on the UK dataset
└── (other modules/files used by the pipeline)
```

---

## Datasets

In this study, we use two datasets:

### **UK dataset**

From:
**Nizzoli, L., Tardelli, S., Avvenuti, M., Cresci, S., & Tesconi, M. (2021, May).**
*Coordinated behavior on social media in 2019 UK general election.*
Proceedings of the International AAAI Conference on Web and Social Media, 15, 443–454.

The **UK dataset is publicly available for research purposes**:
[https://doi.org/10.5281/zenodo.4647893](https://doi.org/10.5281/zenodo.4647893)

---

### **IO Russia dataset**

From:
**Seckin, O. C., Pote, M., Nwala, A. C., Yin, L., Luceri, L., Flammini, A., & Menczer, F. (2025, June).**
*Labeled datasets for research on information operations.*
Proceedings of the International AAAI Conference on Web and Social Media, 19, 2567–2574.

The **IO Russia dataset is publicly available for research purposes**:
[https://zenodo.org/records/14189193](https://zenodo.org/records/14189193)

---

## Configuration Files

### **input_config_IORussia.py**

Contains parameter settings required to run the analysis on the **IO Russia** dataset, including:

* Dataset info
* Preprocessing options
* Model or algorithm parameters

### **input_config_uk.py**

Contains analogous parameters for the **UK** dataset, allowing independent configuration and reproducibility.

---

## Main Execution Scripts

### **main_IORussia.py**

Loads the configuration from `input_config_IORussia.py` and executes the full processing and analysis pipeline for the IO Russia dataset.

Run via:

```bash
python main_IORussia.py
```

### **main_uk.py**

Loads the configuration from `input_config_uk.py` and executes the pipeline for the UK dataset.

Run via:

```bash
python main_uk.py
```

---

## How to Use

1. **Install all dependencies** listed in your environment or requirements file.
2. **Configure parameters** in the relevant `input_config_*.py` file.
3. **Run the corresponding main script** depending on the dataset.
4. Outputs will be stored in `./results/` directory and log will be saved in `./utils/LogManager/log/`

---

## Notes

* Each dataset is handled independently to avoid parameter interference and ensure reproducibility.
* You may duplicate and adapt the current structure to support additional datasets.

---
