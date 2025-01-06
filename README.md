# Full-Glow for Urban Sound Maps

This repository contains an adapted implementation of Full-Glow for urban sound map generation. It is based on [Full-Glow: Fully conditional Glow for more realistic image generation](https://arxiv.org/abs/2012.05846).

The original Full-Glow model has been modified to work with sound map data, allowing for the generation of sound maps from building layouts while considering acoustic and environmental conditions like temperature, humidity and noise levels.

## Environment Setup

1. Create a new virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

2. Install required packages:
```bash
pip install -r requirements.txt
pip install numba  # Required for ray tracing calculations
```

## Project Structure

The project requires the following directory structure:
```
project_root/
├── src/                         # Source code directory
│   ├── main.py                 # Training script
│   ├── generate_soundmaps.py   # Generation script
│   └── sound_metrics.py        # Evaluation script
│
├── data/                       # Base data directory for datasets
│   ├── urban_sound_25k_baseline/
│   ├── urban_sound_25k_diffraction/
│   ├── urban_sound_25k_reflection/
│   └── urban_sound_25k_combined/
│       ├── test/              # Test dataset
│       │   ├── test.csv      # Test set metadata
│       │   ├── buildings/    # Building layouts
│       │   └── soundmaps/    # Ground truth soundmaps
│       └── train/            # Training dataset (similar structure)
│
└── evaluation_results/         # Evaluation outputs
    └── urban_sound_25k_[dataset_type]/
        ├── predictions/       # Generated soundmaps
        └── metrics/          # Evaluation results
```

### Example paths (adjust according to your setup):
- Data Location: `E:/Schallsimulationsdaten/urban_sound_25k_[dataset_type]/`
- Checkpoints: `E:/ba_ergebnisse/urban_sound_25k_[dataset_type]/Checkpoints/`
- Results: `E:/ba_ergebnisse/urban_sound_25k_[dataset_type]/evaluation_results/`

## Training Commands

All training commands use the same base structure. The only difference is the enabled conditions:

### Baseline Model (no conditions):
```bash
python main.py --model glow_improved --dataset soundmap --direction building2soundmap --img_size 256 256 --n_block 4 --n_flow 8 8 8 8 --do_lu --reg_factor 0.0001 --grad_checkpoint
```

### Diffraction Model (with db):
```bash
python main.py --model glow_improved --dataset soundmap --direction building2soundmap --img_size 256 256 --n_block 4 --n_flow 8 8 8 8 --do_lu --reg_factor 0.0001 --grad_checkpoint --use_db
```

### Reflection Model (with db):
```bash
python main.py --model glow_improved --dataset soundmap --direction building2soundmap --img_size 256 256 --n_block 4 --n_flow 8 8 8 8 --do_lu --reg_factor 0.0001 --grad_checkpoint --use_db
```

### Combined Model (all conditions):
```bash
python main.py --model glow_improved --dataset soundmap --direction building2soundmap --img_size 256 256 --n_block 4 --n_flow 8 8 8 8 --do_lu --reg_factor 0.0001 --grad_checkpoint --use_temperature --use_humidity --use_db
```

## Generation and Evaluation

Each dataset requires two steps for evaluation:

1. **Generate Samples**

#### Baseline Model:
```bash
python generate_soundmaps.py --checkpoint_path "E:/ba_ergebnisse/urban_sound_25k_baseline/Checkpoints/soundmap/256x256/glow_improved/building2soundmap" --dataset_path "E:/Schallsimulationsdaten/urban_sound_25k_baseline/test" --output_base "E:/ba_ergebnisse/urban_sound_25k_baseline/evaluation_results" --model_type "urban_sound_25k_baseline"
```

#### Diffraction Model:
```bash
python generate_soundmaps.py --checkpoint_path "E:/ba_ergebnisse/urban_sound_25k_diffraction/Checkpoints/soundmap/256x256/glow_improved/building2soundmap" --dataset_path "E:/Schallsimulationsdaten/urban_sound_25k_diffraction/test" --output_base "E:/ba_ergebnisse/urban_sound_25k_diffraction/evaluation_results" --model_type "urban_sound_25k_diffraction" --use_db
```

#### Reflection Model:
```bash
python generate_soundmaps.py --checkpoint_path "E:/ba_ergebnisse/urban_sound_25k_reflection/Checkpoints/soundmap/256x256/glow_improved/building2soundmap" --dataset_path "E:/Schallsimulationsdaten/urban_sound_25k_reflection/test" --output_base "E:/ba_ergebnisse/urban_sound_25k_reflection/evaluation_results" --model_type "urban_sound_25k_reflection" --use_db
```

#### Combined Model:
```bash
python generate_soundmaps.py --checkpoint_path "E:/ba_ergebnisse/urban_sound_25k_combined/Checkpoints/soundmap/256x256/glow_improved/building2soundmap" --dataset_path "E:/Schallsimulationsdaten/urban_sound_25k_combined/test" --output_base "E:/ba_ergebnisse/urban_sound_25k_combined/evaluation_results" --model_type "urban_sound_25k_combined" --use_temperature --use_humidity --use_db
```

2. **Calculate Metrics**

#### Baseline Model:
```bash
python sound_metrics.py --data_dir "E:/Schallsimulationsdaten/urban_sound_25k_baseline/test" --pred_dir "E:/ba_ergebnisse/urban_sound_25k_baseline/evaluation_results/urban_sound_25k_baseline/predictions" --output_dir "E:/ba_ergebnisse/urban_sound_25k_baseline/evaluation_results" --model_type "urban_sound_25k_baseline"
```

#### Diffraction Model:
```bash
python sound_metrics.py --data_dir "E:/Schallsimulationsdaten/urban_sound_25k_diffraction/test" --pred_dir "E:/ba_ergebnisse/urban_sound_25k_diffraction/evaluation_results/urban_sound_25k_diffraction/predictions" --output_dir "E:/ba_ergebnisse/urban_sound_25k_diffraction/evaluation_results" --model_type "urban_sound_25k_diffraction"
```

#### Reflection Model:
```bash
python sound_metrics.py --data_dir "E:/Schallsimulationsdaten/urban_sound_25k_reflection/test" --pred_dir "E:/ba_ergebnisse/urban_sound_25k_reflection/evaluation_results/urban_sound_25k_reflection/predictions" --output_dir "E:/ba_ergebnisse/urban_sound_25k_reflection/evaluation_results" --model_type "urban_sound_25k_reflection"
```

#### Combined Model:
```bash
python sound_metrics.py --data_dir "E:/Schallsimulationsdaten/urban_sound_25k_combined/test" --pred_dir "E:/ba_ergebnisse/urban_sound_25k_combined/evaluation_results/urban_sound_25k_combined/predictions" --output_dir "E:/ba_ergebnisse/urban_sound_25k_combined/evaluation_results" --model_type "urban_sound_25k_combined"
```

## Evaluation Metrics

The evaluation script calculates several metrics:
- **MAE (Mean Absolute Error):** Average absolute difference between predicted and true values
- **MAPE (Mean Absolute Percentage Error):** Average percentage difference between predicted and true values

### Line of Sight (LoS) Metrics:
- **LoS_MAE:** MAE for areas with direct line of sight to the sound source
- **NLoS_MAE:** MAE for areas without direct line of sight
- **LoS_wMAPE:** Weighted MAPE for visible areas
- **NLoS_wMAPE:** Weighted MAPE for non-visible areas

## Output Structure

Evaluation results are organized as follows:
```
evaluation_results/
└── urban_sound_25k_[dataset_type]/
    ├── predictions/          # Generated soundmaps
    ├── metrics/             # Evaluation results
    │   ├── detailed_results.csv         # Per-sample metrics
    │   ├── summary_statistics.csv       # Statistical overview
    │   └── plots/                      # Metric visualizations
    └── logs/                # Error logs
```

## Required CSV Format

The `test.csv` file must contain:
```csv
osm,soundmap,temperature,humidity,db
buildings/001.png,soundmaps/001.png,25.5,65.0,{"lwd500": 75.2}
buildings/002.png,soundmaps/002.png,23.8,70.0,{"lwd500": 68.4}
```

### Note:
- `db` values are JSON strings with "lwd500" key
- All paths are relative to the data directory

### Training Parameters

- `--n_flow`: Number of flow steps per block (default: [8, 8, 8, 8])
- `--n_block`: Number of blocks (default: 4) 
- `--img_size`: Image dimensions (default: [256, 256])
- `--batch_size`: Batch size for training (default: 1)
- `--lr`: Learning rate (default: 1e-4)
- `--temperature`: Sampling temperature (default: 1.0)
- `--do_lu`: Enable LU decomposition for invertible 1x1 convolutions
- `--grad_checkpoint`: Enable gradient checkpointing to reduce memory usage
- `--checkpoint_reentrant`: Set checkpoint reentrant behavior (default: True)
- `--use_temperature`: Enable temperature conditioning
- `--use_humidity`: Enable humidity conditioning
- `--use_db`: Enable decibel level conditioning

## Model Architecture

The model uses the Full-Glow architecture which extends the original Glow model by making all operations conditional. When environmental conditions are enabled, they are incorporated into the conditioning networks alongside the building layout information.

Key features:
- Fully conditional flow-based model
- Support for multiple environmental conditions
- Support for combining image and numerical conditions
- Generates 256x256 sound maps
- Uses LU decomposition for improved stability (when enabled)
- Gradient checkpointing for memory efficiency

## Citation

If you use this code, please cite both this adaptation and the original Full-Glow paper:

```
@inproceedings{sorkhei2021full,
  author={Sorkhei, Moein and Henter, Gustav Eje and Kjellstr{\"o}m, Hedvig},
  title={Full-{G}low: {F}ully conditional {G}low for more realistic image generation},
  booktitle={Proceedings of the DAGM German Conference on Pattern Recognition (GCPR)},
  volume={43},
  month={Oct.},
  year={2021}
}
```

## License
MIT