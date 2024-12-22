# Full-Glow for Urban Sound Maps

This repository contains an adapted implementation of Full-Glow for urban sound map generation. It is based on [Full-Glow: Fully conditional Glow for more realistic image generation](https://arxiv.org/abs/2012.05846).

The original Full-Glow model has been modified to work with sound map data, allowing for the generation of sound maps from building layouts while optionally considering environmental conditions like temperature, humidity and sound levels.

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

## Training Commands

### Basic Training
Train the model using only building layouts to generate sound maps:
```bash
python main.py --dataset soundmap --direction building2soundmap --model glow_improved
```

### Training with Environmental Conditions
You can include different combinations of environmental conditions:

1. Using temperature only:
```bash
python main.py --dataset soundmap --direction building2soundmap --model glow_improved --use_temperature
```

2. Using humidity only:
```bash
python main.py --dataset soundmap --direction building2soundmap --model glow_improved --use_humidity
```

3. Using decibel level only:
```bash
python main.py --dataset soundmap --direction building2soundmap --model glow_improved --use_db
```

4. Using multiple conditions:
```bash
python main.py --dataset soundmap --direction building2soundmap --model glow_improved --use_temperature --use_humidity --use_db
```

### Additional Training Parameters

- `--n_flow`: Number of flow steps per block (default: [32, 32, 32, 32])
- `--n_block`: Number of blocks (default: 4) 
- `--img_size`: Image dimensions (default: [256, 256])
- `--batch_size`: Batch size for training (default: 1)
- `--lr`: Learning rate (default: 1e-4)
- `--temperature`: Sampling temperature (default: 1.0)
- `--do_lu`: Enable LU decomposition for invertible 1x1 convolutions
- `--grad_checkpoint`: Enable gradient checkpointing to reduce memory usage

Recommended:
```bash
python main.py --model glow_improved --dataset soundmap --direction building2soundmap --img_size 256 256 --n_block 4 --n_flow 8 8 8 8 --do_lu --reg_factor 0.0001 --grad_checkpoint --use_temperature --use_humidity --use_db
```

## Model Architecture

The model uses the Full-Glow architecture which extends the original Glow model by making all operations conditional. When environmental conditions are enabled, they are incorporated into the conditioning networks alongside the building layout information.

Key features:
- Fully conditional flow-based model
- Support for multiple environmental conditions
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
