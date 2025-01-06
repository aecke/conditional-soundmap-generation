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
```

## Data Structure

The model expects the following data structure:
- Training data CSV containing building-soundmap pairs and environmental conditions
- Building images
- Sound map images (256x256 pixels)

Configure the data paths in `params.json`:
```json
"soundmap": {
    "data_folder": {
        "train": {
            "buildings": "path/to/train/buildings",
            "soundmaps": "path/to/train/soundmaps/256",
            "csv_path": "path/to/train.csv"
        },
        "test": {
            "buildings": "path/to/test/buildings", 
            "soundmaps": "path/to/test/soundmaps/256",
            "csv_path": "path/to/test.csv"
        }
    },
    "samples_path": "path/to/samples",
    "checkpoints_path": "path/to/checkpoints"
}
```
## Training, Generation and Evaluation

### 1. Training Commands

### Dataset-specific Training Commands

1. Baseline Model (without conditions):
```bash
python main.py --model glow_improved --dataset soundmap --direction building2soundmap --img_size 256 256 --n_block 4 --n_flow 8 8 8 8 --do_lu --reg_factor 0.0001 --grad_checkpoint --checkpoint_reentrant False
```

2. Diffraction Model (with noise level):
```bash
python main.py --model glow_improved --dataset soundmap --direction building2soundmap --img_size 256 256 --n_block 4 --n_flow 8 8 8 8 --do_lu --reg_factor 0.0001 --grad_checkpoint --use_db --checkpoint_reentrant False
```

3. Reflection Model (with noise level):
```bash
python main.py --model glow_improved --dataset soundmap --direction building2soundmap --img_size 256 256 --n_block 4 --n_flow 8 8 8 8 --do_lu --reg_factor 0.0001 --grad_checkpoint --use_db --checkpoint_reentrant False
```

4. Combined Model (with all conditions):
```bash
python main.py --model glow_improved --dataset soundmap --direction building2soundmap --img_size 256 256 --n_block 4 --n_flow 8 8 8 8 --do_lu --reg_factor 0.0001 --grad_checkpoint --use_temperature --use_humidity --use_db --checkpoint_reentrant False
```

### 2. Sample Generation

After training, you can generate samples and evaluate the model using the following commands for each dataset:

#### Baseline Model:
```bash
python generate_soundmaps.py \
  --checkpoint_path "path/to/urban_sound_25k_baseline/Checkpoints/soundmap/256x256/glow_improved/building2soundmap" \
  --dataset_path "path/to/urban_sound_25k_baseline/test" \
  --output_base "path/to/evaluation_results" \
  --model_type "urban_sound_25k_baseline"
```

#### Diffraction Model:
```bash
python generate_soundmaps.py \
  --checkpoint_path "path/to/urban_sound_25k_diffraction/Checkpoints/soundmap/256x256/glow_improved/building2soundmap" \
  --dataset_path "path/to/urban_sound_25k_diffraction/test" \
  --output_base "path/to/evaluation_results" \
  --model_type "urban_sound_25k_diffraction" \
  --use_db
```

#### Reflection Model:
```bash
python generate_soundmaps.py \
  --checkpoint_path "path/to/urban_sound_25k_reflection/Checkpoints/soundmap/256x256/glow_improved/building2soundmap" \
  --dataset_path "path/to/urban_sound_25k_reflection/test" \
  --output_base "path/to/evaluation_results" \
  --model_type "urban_sound_25k_reflection" \
  --use_db
```

#### Combined Model:
```bash
python generate_soundmaps.py \
  --checkpoint_path "path/to/urban_sound_25k_combined/Checkpoints/soundmap/256x256/glow_improved/building2soundmap" \
  --dataset_path "path/to/urban_sound_25k_combined/test" \
  --output_base "path/to/evaluation_results" \
  --model_type "urban_sound_25k_combined" \
  --use_temperature --use_humidity --use_db
```

### 3. Model Evaluation

After generating samples, run the evaluation script to calculate metrics. Replace TIMESTAMP with the timestamp of your generation run (format: YYYYMMDD_HHMMSS):

#### Baseline Model:
```bash
python sound_metrics.py \
  --data_dir "path/to/urban_sound_25k_baseline/test" \
  --pred_dir "path/to/evaluation_results/urban_sound_25k_baseline/predictions" \
  --output_dir "path/to/evaluation_results" \
  --model_type "urban_sound_25k_baseline"
```

#### Diffraction Model:
```bash
python sound_metrics.py \
  --data_dir "path/to/urban_sound_25k_diffraction/test" \
  --pred_dir "path/to/evaluation_results/urban_sound_25k_diffraction/predictions" \
  --output_dir "path/to/evaluation_results" \
  --model_type "urban_sound_25k_diffraction"
```

#### Reflection Model:
```bash
python sound_metrics.py \
  --data_dir "path/to/urban_sound_25k_reflection/test" \
  --pred_dir "path/to/evaluation_results/urban_sound_25k_reflection/predictions" \
  --output_dir "path/to/evaluation_results" \
  --model_type "urban_sound_25k_reflection"
```

#### Combined Model:
```bash
python sound_metrics.py \
  --data_dir "path/to/urban_sound_25k_combined/test" \
  --pred_dir "path/to/evaluation_results/urban_sound_25k_combined/predictions" \
  --output_dir "path/to/evaluation_results" \
  --model_type "urban_sound_25k_combined"
```

### Evaluation Metrics

The evaluation script calculates the following metrics:
- MAE (Mean Absolute Error)
- MAPE (Mean Absolute Percentage Error)
- LoS (Line of Sight) specific metrics
  - LoS_MAE: MAE for visible areas
  - NLoS_MAE: MAE for non-visible areas
  - LoS_wMAPE: Weighted MAPE for visible areas
  - NLoS_wMAPE: Weighted MAPE for non-visible areas

### Output Structure

The evaluation creates the following directory structure:
```
evaluation_results/
├── urban_sound_25k_baseline/
├── urban_sound_25k_diffraction/
├── urban_sound_25k_combined/
└── urban_sound_25k_reflection/
    └── TIMESTAMP/
        ├── predictions/          # Generated soundmaps
        ├── metrics/             # Evaluation results
        │   ├── inference_times.csv
        │   ├── performance_statistics.json
        │   ├── detailed_results.csv
        │   ├── summary_statistics.csv
        │   └── plots/          # Visualization plots
        ├── logs/               # Error logs
        └── evaluation_config.json


### CSV Format for Environmental Conditions
When using environmental conditions, your CSV file should contain the following columns:
- `osm`: Path to building image
- `soundmap`: Path to soundmap image
- `temperature`: Temperature value (if using --use_temperature)
- `humidity`: Humidity value (if using --use_humidity)
- `db`: Decibel value (if using --use_db)

Example CSV format:
```csv
osm,soundmap,temperature,humidity,db
buildings/001.png,soundmaps/001.png,25.5,65.0,{"lwd500": 75.2}
buildings/002.png,soundmaps/002.png,23.8,70.0,{"lwd500": 68.4}
```

Note: The db values are stored as JSON strings containing the "lwd500" key.

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