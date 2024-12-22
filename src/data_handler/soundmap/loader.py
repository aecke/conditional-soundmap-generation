# data_handler/soundmap/loader.py
from torch.utils import data
from torchvision import transforms
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
        
        # CSV einlesen
        self.data = pd.read_csv(data_paths["csv_path"])
        print(f"Found {len(self.data)} entries in {'train' if is_train else 'test'} set")

    def __len__(self):  # Diese Methode muss auf der ersten Ebene der Klasse sein
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
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

        numerical_conditions = []
        if self.use_db:
            db_value = eval(row['db'])['lwd500']
            numerical_conditions.append(db_value)
        if self.use_temperature:
            numerical_conditions.append(row['temperature'])
        if self.use_humidity:
            numerical_conditions.append(row['humidity'])

        if numerical_conditions:
            result['numerical_conditions'] = torch.tensor(numerical_conditions, dtype=torch.float32)

        return result