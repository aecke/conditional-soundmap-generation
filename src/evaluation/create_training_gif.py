import os
import glob
from PIL import Image
import argparse
from tqdm import tqdm
import re

def natural_sort_key(s):
    """Schlüsselfunktion für natürliche Sortierung von Strings mit Zahlen"""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split('([0-9]+)', s)]

def create_training_gif(samples_dir, output_path, duration=500, max_size=None):
    """
    Erstellt ein GIF aus den Trainings-Samples
    
    Args:
        samples_dir: Verzeichnis mit den PNG-Dateien
        output_path: Pfad für das resultierende GIF
        duration: Dauer pro Frame in Millisekunden
        max_size: Maximale Breite/Höhe für das GIF (behält Seitenverhältnis bei)
    """
    # Finde alle PNG-Dateien und sortiere sie numerisch nach der Iterationsnummer
    png_files = glob.glob(os.path.join(samples_dir, "*.png"))
    png_files = sorted(png_files, key=natural_sort_key)
    
    if not png_files:
        raise ValueError(f"Keine PNG-Dateien gefunden in {samples_dir}")
        
    print(f"Gefundene PNG-Dateien: {len(png_files)}")
    
    # Liste für die Frames
    frames = []
    
    # Verarbeite jedes Bild
    for png_file in tqdm(png_files, desc="Verarbeite Bilder"):
        img = Image.open(png_file)
        
        # Resizing wenn max_size angegeben
        if max_size:
            ratio = min(max_size/img.width, max_size/img.height)
            new_size = (int(img.width * ratio), int(img.height * ratio))
            img = img.resize(new_size, Image.Resampling.LANCZOS)
        
        # Konvertiere zu RGB falls notwendig
        if img.mode != 'RGB':
            img = img.convert('RGB')
            
        frames.append(img)
    
    if not frames:
        raise ValueError("Keine Bilder gefunden zum Verarbeiten")
    
    # Erstelle das GIF
    print(f"Erstelle GIF mit {len(frames)} Frames...")
    frames[0].save(
        output_path,
        save_all=True,
        append_images=frames[1:],
        duration=duration,
        loop=0,
        optimize=True
    )
    print(f"GIF gespeichert als: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Erstellt ein GIF aus Trainings-Samples")
    parser.add_argument("--samples_dir", type=str, required=True,
                      help="Verzeichnis mit den PNG-Dateien")
    parser.add_argument("--output_path", type=str, required=True,
                      help="Pfad für das resultierende GIF")
    parser.add_argument("--duration", type=int, default=500,
                      help="Dauer pro Frame in Millisekunden (default: 500)")
    parser.add_argument("--max_size", type=int, default=256,
                      help="Maximale Breite/Höhe für das GIF (default: 256)")
    
    args = parser.parse_args()
    create_training_gif(args.samples_dir, args.output_path, args.duration, args.max_size)