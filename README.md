# Semantic Caching for Low-Cost LLM Serving
Unofficial reproduction of semantic caching algorithms for low-cost LLM serving, for INFOCOM 2026.

## Overview
This repository contains experiment codes to evaluate semantic caching strategies for LLM inference, including offline precomputation, online serving and ablation studies.

## File Structure
- `experiment_offline.py`: Offline semantic cache construction
- `experiment_online.py`: Online LLM serving with semantic cache
- `experiment_online_ablations.py`: Ablation experiments
- `experiment_known_parameters.py`: Experiments with known cache parameters
- `queries20.txt`: Test query set
- `*_README.md`: Detailed notes for each experiment group

## Requirements
Python 3.8+

## Run
```bash
python experiment_offline.py
python experiment_online.py
