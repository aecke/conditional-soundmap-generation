import torch
from torch.utils import data
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
import joblib
from PIL import Image
import pandas as pd
import os
import glob

class SoundMapDataset(data.Dataset):
    SCALER_FILENAME = 'extra_cond_scaler.save'
    # test
    def __init__(self, data_folder, is_train=True, img_size=(256, 256), 
                 use_temperature=False, use_humidity=False, use_db=False, min_pixel_value=0.01):
        self.min_pixel_value = min_pixel_value
        # Custom normalization transform
        self.transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: self._normalize_tensor(x))  # Custom normalization
        ])
        
        self.img_size = img_size
        self.use_temperature = use_temperature
        self.use_humidity = use_humidity
        self.use_db = use_db
        
        # Wähle train oder test Pfade
        data_paths = data_folder["train"] if is_train else data_folder["test"]
        self.buildings_path = data_paths["buildings"]
        self.soundmaps_path = data_paths["soundmaps"]
        self.is_train = is_train
        self.scaler_path = os.path.join(os.path.dirname(data_folder["train"]["csv_path"]), self.SCALER_FILENAME)

        # CSV einlesen
        self.data = pd.read_csv(data_paths["csv_path"])
        
        # DB-Werte aus dict extrahieren
        if self.use_db:
            try:
                self.data['db_value'] = self.data['db'].apply(lambda x: eval(x)['lwd500'])
            except (KeyError, SyntaxError) as e:
                print("Error extracting db values from dictionary")
                raise ValueError(f"Invalid db value format: {e}")
        
        # Liste der zu normalisierenden Spalten erstellen
        self.extra_cond_cols = []
        if self.use_db:
            self.extra_cond_cols.append('db_value')
        if self.use_temperature:
            self.extra_cond_cols.append('temperature')
        if self.use_humidity:
            self.extra_cond_cols.append('humidity')
            
        # Scaler initialisieren und anwenden wenn numerische Features verwendet werden
        if self.extra_cond_cols and is_train:
            self.scaler = MinMaxScaler()
            self.data[self.extra_cond_cols] = self.scaler.fit_transform(self.data[self.extra_cond_cols])
            # Speichere Scaler für Testset
            joblib.dump(self.scaler, self.scaler_path)
            
            print(f"Using extra_cond conditions: {', '.join(self.extra_cond_cols)}")
            print(f"Fitted scaler saved to: {self.scaler_path}")
            print("extra_cond ranges after scaling:")
            for col in self.extra_cond_cols:
                min_val = self.data[col].min()
                max_val = self.data[col].max()
                print(f"  {col}: [{min_val:.3f}, {max_val:.3f}]")
                
        elif self.extra_cond_cols:
            # Lade gespeicherten Scaler für Testset
            self.scaler = joblib.load(self.scaler_path)
            self.data[self.extra_cond_cols] = self.scaler.transform(self.data[self.extra_cond_cols])
            
        print(f"Found {len(self.data)} entries in {'train' if is_train else 'test'} set")
    
    def _normalize_tensor(self, x):
        """Custom normalization to ensure minimum value"""
        # Rescale from [0,1] to [min_value,1]
        return x * (1 - self.min_pixel_value) + self.min_pixel_value
    
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Pfade erstellen
        building_name = os.path.basename(row['osm'])
        soundmap_name = os.path.basename(row['soundmap'])
        
        building_path = os.path.join(self.buildings_path, building_name)
        soundmap_path = os.path.join(self.soundmaps_path, soundmap_name)

        # Prüfe ob Dateien existieren
        if not os.path.exists(building_path):
            raise FileNotFoundError(f"Building file not found: {building_path}")
        if not os.path.exists(soundmap_path):
            raise FileNotFoundError(f"Soundmap file not found: {soundmap_path}")

        # Bilder laden
        building = self.transforms(Image.open(building_path))
        soundmap = self.transforms(Image.open(soundmap_path))
        
        # Validiere Bildgrößen
        for name, img in [("building", building), ("soundmap", soundmap)]:
            if img.size(1) != self.img_size[0] or img.size(2) != self.img_size[1]:
                raise ValueError(f"Unexpected {name} size: got {img.size()}, expected {(3,) + self.img_size}")
        
        if building.size(0) == 1:
            building = building.repeat(3, 1, 1)
        if soundmap.size(0) == 1:
            soundmap = soundmap.repeat(3, 1, 1)

        result = {
            'building': building,
            'soundmap': soundmap,
            'building_path': building_path,
            'soundmap_path': soundmap_path
        }

        # Numerische Bedingungen sammeln (bereits normalisiert)
        if self.extra_cond_cols:
            try:
                extra_cond = []
                for col in self.extra_cond_cols:
                    if pd.isna(row[col]):
                        raise ValueError(f"Missing value for {col} in row {idx}")
                    extra_cond.append(float(row[col]))  # Ensure float values
                    
                # Create tensor with explicit shape
                extra_cond = torch.tensor(extra_cond, dtype=torch.float32)
                print(f"Dataset extra_cond conditions shape: {extra_cond.shape}")  # Should be (n_features,)
                result['extra_cond'] = extra_cond
                
            except Exception as e:
                print(f"Error processing extra_cond conditions for row {idx}: {e}")
                print(f"Problematic row data: {row[self.extra_cond_cols]}")
                raise
        return result