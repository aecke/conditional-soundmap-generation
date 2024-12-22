# data_handler/soundmap/loader.py
from torch.utils import data
from torchvision import transforms
from sklearn.preprocessing import MinMaxScaler
from sklearn.externals import joblib
from PIL import Image
import pandas as pd
import os
import glob

class SoundMapDataset(data.Dataset):
    def __init__(self, data_folder, is_train=True, img_size=(512, 512), 
                 use_temperature=False, use_humidity=False, use_db=False):
        self.transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
        
        self.use_temperature = use_temperature
        self.use_humidity = use_humidity
        self.use_db = use_db
        
        # Wähle train oder test Pfade
        data_paths = data_folder["train"] if is_train else data_folder["test"]
        self.buildings_path = data_paths["buildings"]
        self.soundmaps_path = data_paths["soundmaps"]
        self.is_train = is_train
        
        # CSV einlesen
        self.data = pd.read_csv(data_paths["csv_path"])
        
        # DB-Werte aus dict extrahieren
        if self.use_db:
            self.data['db_value'] = self.data['db'].apply(lambda x: eval(x)['lwd500'])
        
        # Liste der zu normalisierenden Spalten erstellen
        self.numerical_cols = []
        if self.use_db:
            self.numerical_cols.append('db_value')
        if self.use_temperature:
            self.numerical_cols.append('temperature')
        if self.use_humidity:
            self.numerical_cols.append('humidity')
            
        # Scaler initialisieren und anwenden wenn numerische Features verwendet werden
        if self.numerical_cols and is_train:
            self.scaler = MinMaxScaler()
            self.data[self.numerical_cols] = self.scaler.fit_transform(self.data[self.numerical_cols])
            # Speichere Scaler für Testset
            joblib.dump(self.scaler, os.path.join(data_folder["train"], 'scaler.save'))
        elif self.numerical_cols:
            # Lade gespeicherten Scaler für Testset
            self.scaler = joblib.load(os.path.join(data_folder["train"], 'scaler.save'))
            self.data[self.numerical_cols] = self.scaler.transform(self.data[self.numerical_cols])
            
        print(f"Found {len(self.data)} entries in {'train' if is_train else 'test'} set")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # Pfade erstellen
        building_name = os.path.basename(row['osm'])
        soundmap_name = os.path.basename(row['soundmap'])
        
        building_path = os.path.join(self.buildings_path, building_name)
        soundmap_path = os.path.join(self.soundmaps_path, soundmap_name)

        # Bilder laden
        building = self.transforms(Image.open(building_path))
        soundmap = self.transforms(Image.open(soundmap_path))
        
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
        if self.numerical_cols:
            numerical_conditions = []
            for col in self.numerical_cols:
                numerical_conditions.append(row[col])
            result['numerical_conditions'] = torch.tensor(numerical_conditions, dtype=torch.float32)

        return result