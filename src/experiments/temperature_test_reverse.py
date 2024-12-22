import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from torchvision import utils
import helper
import models
import data_handler
from globals import device
from PIL import Image
from torchvision import transforms
import glob
import matplotlib.pyplot as plt
import numpy as np

def create_summary_plot(original_soundmap, samples_dict, output_path, sample_name):
    """Erstellt eine Übersichtsgrafik mit Original und Samples für alle Temperaturen."""
    
    # Konvertiere torch tensors zu numpy arrays und normalisiere
    def tensor_to_numpy(tensor):
        if tensor.is_cuda:
            tensor = tensor.cpu()
        array = tensor.numpy()
        array = np.transpose(array, (1, 2, 0))  # CHW -> HWC
        array = (array - array.min()) / (array.max() - array.min())  # Normalisierung auf [0,1]
        return array
    
    temperatures = sorted(samples_dict.keys(), reverse=True)
    n_temps = len(temperatures)
    n_samples = len(samples_dict[temperatures[0]])
    
    # Erstelle Figure und Grid
    fig = plt.figure(figsize=(20, 3*n_temps))
    gs = plt.GridSpec(n_temps, n_samples+1, figure=fig)
    
    # Plot original soundmap in der ersten Spalte jeder Zeile
    original_img = tensor_to_numpy(original_soundmap)
    
    # Titel für die gesamte Figur
    plt.suptitle(f'Temperature Sampling Results for {sample_name} (Soundmap to Building)', size=16, y=0.95)
    
    # Beschriftungen für die Spalten
    for i in range(n_samples):
        plt.figtext(0.2 + (i+1)*0.8/n_samples, 0.92, f'Sample {i+1}', 
                   ha='center', va='bottom')
    plt.figtext(0.1, 0.92, 'Original\nSoundmap', ha='center', va='bottom')
    
    for i, temp in enumerate(temperatures):
        # Original in erster Spalte
        ax = fig.add_subplot(gs[i, 0])
        ax.imshow(original_img)
        ax.set_title(f'T={temp:.1f}')
        ax.axis('off')
        
        # Samples in weiteren Spalten
        for j, sample in enumerate(samples_dict[temp]):
            ax = fig.add_subplot(gs[i, j+1])
            sample_img = tensor_to_numpy(sample)
            ax.imshow(sample_img)
            ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 1, 0.9])  # Anpassung für den Haupttitel
    
    # Speichern
    plt.savefig(os.path.join(output_path, f'{sample_name}_summary.png'), 
                bbox_inches='tight', dpi=300)
    plt.close()

def sample_with_temperatures(args, params, checkpoint_path, output_base_path, soundmap_index=0):
    # Model initialisieren und Checkpoint laden
    model = models.init_model(args, params)
    model = helper.load_checkpoint(
        checkpoint_path, 
        100000,  # Checkpoint von 100k iterations
        model, 
        None,
        resume_train=False
    )[0]
    
    # Alle Test-Soundmaps finden und sortieren
    test_soundmaps = sorted(glob.glob(os.path.join(params['data_folder']['test']['soundmaps'], '*_LEQ_512.png')))
    
    # Gewählte Soundmap laden
    test_soundmap_path = test_soundmaps[soundmap_index]
    soundmap_name = os.path.basename(test_soundmap_path).split('.')[0]
    
    print(f"Using soundmap: {soundmap_name}")
    
    transform = transforms.Compose([
        transforms.Resize(params['img_size']),
        transforms.ToTensor(),
    ])
    test_soundmap = transform(Image.open(test_soundmap_path))
    
    # Auf 3 Kanäle erweitern falls nötig
    if test_soundmap.size(0) == 1:
        test_soundmap = test_soundmap.repeat(3, 1, 1)
    
    test_soundmap = test_soundmap.unsqueeze(0).to(device)  # Batch-Dimension hinzufügen
    
    # Output Basisverzeichnis mit Soundmap-Name
    output_base_path = os.path.join(output_base_path, soundmap_name)
    helper.make_dir_if_not_exists(output_base_path)
    
    # Für verschiedene Temperaturen samplen
    temperatures = [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.0]
    
    # Original Soundmap speichern
    utils.save_image(
        test_soundmap[0],
        os.path.join(output_base_path, 'original_soundmap.png'),
        normalize=True
    )
    
    # Dictionary für alle Samples
    all_samples = {}
    
    for temp in temperatures:
        print(f"Generating samples with temperature {temp}")
        
        # Output Verzeichnis für diese Temperatur
        temp_dir = os.path.join(output_base_path, f"temp_{temp:.1f}")
        helper.make_dir_if_not_exists(temp_dir)
        
        # Temperature setzen
        params['temperature'] = temp
        
        # 5 Samples generieren
        with torch.no_grad():
            sampled_images = models.take_samples(
                args, 
                params,
                model, 
                test_soundmap,  # Die Test-Soundmap als condition
                n_samples=5
            )
            
            # Speichere Samples für Summary Plot
            all_samples[temp] = sampled_images
            
            # Grid aus allen Samples erstellen
            utils.save_image(
                sampled_images,
                os.path.join(temp_dir, f'samples_grid.png'),
                nrow=5,
                normalize=True
            )
            
            # Einzelne Samples speichern
            for i, sample in enumerate(sampled_images):
                utils.save_image(
                    sample,
                    os.path.join(temp_dir, f'sample_{i+1}.png'),
                    normalize=True
                )
    
    # Erstelle Summary Plot
    create_summary_plot(test_soundmap[0], all_samples, output_base_path, soundmap_name)

if __name__ == "__main__":
    # Pfade anpassen
    CHECKPOINT_PATH = "E:/Schallsimulationsdaten/urban_sound_25k_reflection/newtrainingsparameterCheckpoints/soundmap/512x512/glow_improved/soundmap2building"  # Angepasster Pfad!
    OUTPUT_BASE_PATH = "E:/Schallsimulationsdaten/urban_sound_25k_reflection/temperature_samples_100k_reverse"  # Neuer Output-Pfad
    
    # Args und params aus dem Training verwenden
    from argparse import Namespace
    args = Namespace(
        dataset="soundmap",
        direction="soundmap2building",  # Umgekehrte Richtung!
        model="glow_improved",
        do_lu=True,
        grad_checkpoint=True,
        n_block=4,
        n_flow=[8, 8, 8, 8],
        reg_factor=0.0001,
        use_bmaps=False,
        exp=False
    )
    
    # Params aus params.json
    params = {
        'sample_freq': 500,
        'checkpoint_freq': 1000,
        'val_freq': 1000,
        'iter': 100000,
        'n_flow': [8, 8, 8, 8],
        'n_block': 4,
        'lu': True,
        'affine': True,
        'n_bits': 8,
        'lr': 0.0001,
        'temperature': 1.0,  # wird später überschrieben
        'n_samples': 5,
        'channels': 3,
        'img_size': [512, 512],
        'batch_size': 1,
        'monitor_val': True,
        'data_folder': {
            'train': {
                'buildings': 'E:/Schallsimulationsdaten/urban_sound_25k_reflection/train/buildings',
                'soundmaps': 'E:/Schallsimulationsdaten/urban_sound_25k_reflection/train/soundmaps/512'
            },
            'test': {
                'buildings': 'E:/Schallsimulationsdaten/urban_sound_25k_reflection/test/buildings',
                'soundmaps': 'E:/Schallsimulationsdaten/urban_sound_25k_reflection/test/soundmaps/512'
            }
        }
    }
    
    # Sample für die erste Soundmap (Index 0) generieren
    print("Generating samples for first test soundmap...")
    sample_with_temperatures(args, params, CHECKPOINT_PATH, OUTPUT_BASE_PATH, soundmap_index=0)
    
    # Sample für die 61te Soundmap (Index 60) generieren
    print("\nGenerating samples for 61st test soundmap...")
    sample_with_temperatures(args, params, CHECKPOINT_PATH, OUTPUT_BASE_PATH, soundmap_index=60)