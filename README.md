# Attribute Inference Attacks

This repository contains implementations of various attribute inference attacks on machine learning models. Based on ART from [[https://github.com/Trusted-AI/adversarial-robustness-toolbox]]

## Overview

Attribute inference attacks aim to infer sensitive attributes of data that were intentionally excluded from the training process. This project demonstrates how such attacks work and visualizes the information leakage in machine learning models.

## Structure

- **Main Notebook**: [attribute_inference.ipynb](attribute_inference.ipynb) - The primary entry point for running and exploring the attacks
- **Source Code**: All implementation code is organized in the `src` directory:
  - `src/attacks/` - Implementation of baseline, black-box, and white-box attacks
  - `src/data/` - Data loading and preprocessing utilities
  - `src/models/` - Target model implementations
  - `src/utils/` - Evaluation and visualization utilities

## Getting Started

1. Install the required dependencies:
   ```
   pip install -r requirements.txt
   ```

2. Open and run the main notebook:
   ```
   jupyter notebook attribute_inference.ipynb
   ```

## License

[MIT](LICENSE)
