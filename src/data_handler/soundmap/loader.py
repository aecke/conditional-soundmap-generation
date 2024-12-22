# data_handler/soundmap/loader.py
from torch.utils import data
from torchvision import transforms
from PIL import Image
import os
import glob

class SoundMapDataset(data.Dataset):
    def __init__(self, data_folder, is_train=True, img_size=(512, 512)):
        self.transforms = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
        ])
        self.use_temperature = use_temperature
        self.use_humidity = use_humidity
        self.use_db= use_db
        
        csv_path = os.path.join(data_folder['base' if is_train else 'test'], "train.csv" if is_train else "test.csv")
        self.data = pd.read_csv(csv_path)
        
        # Wähle train oder test Pfade
        if is_train:
            buildings_path = data_folder["train"]["buildings"]
            soundmaps_path = data_folder["train"]["soundmaps"]
        else:
            buildings_path = data_folder["test"]["buildings"]
            soundmaps_path = data_folder["test"]["soundmaps"]
        
        # Get all building files
        self.building_files = glob.glob(os.path.join(buildings_path, "osm_*.png"))
        
        # Create dictionary to map building files to soundmap files
        self.pairs = {}
        for building_file in self.building_files:
            building_num = building_file.split("osm_")[-1].split(".")[0]
            soundmap_file = os.path.join(soundmaps_path, f"{building_num}_LEQ_512.png")
            if os.path.exists(soundmap_file):
                self.pairs[building_file] = soundmap_file
                
        print(f"Found {len(self.pairs)} matching pairs in {'train' if is_train else 'test'} set")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        building_path = os.path.join(self.base_path, row['osm'].lstrip('./'))
        soundmap_path = os.path.join(self.base_path, row['soundmap_256'].lstrip('./'))

        building = self.transforms(Image.open(building_path))
        soundmap = self.transforms(Image.open(soundmap_path))
        
        if building.size(0) == 1:
            building = building.repeat(3, 1, 1)
        if soundmap.size(0) == 1:
            soundmap = soundmap.repeat(3, 1, 1)

        numerical_conditions = []
        if self.use_temperature:
            numerical_conditions.append(row['temperature'])
        if self.use_humidity:
            numerical_conditions.append(row['humidity'])
        if self.use_db:
            numerical_conditions.append(row['db'])

        result = {
            'building': building,
            'soundmap': soundmap,
            'building_path': building_path,
            'soundmap_path': soundmap_path,
        }

        if numerical_conditions:
            result['numerical_conditions'] = torch.tensor(numerical_conditions, dtype=torch.float32)

        return result

def init_soundmap_loaders(args, params):
    dataset_params = {
        'data_folder': params['data_folder'],
        'img_size': params['img_size'],
        'use_temperature': args.use_temperature,
        'use_humidity': args.use_humidity
    }

    loader_params = {
        "batch_size": params['batch_size'],
        'num_workers': 0
    }

    train_dataset = SoundMapDataset(**dataset_params, is_train=True)
    val_dataset = SoundMapDataset(**dataset_params, is_train=False)

    train_loader = DataLoader(train_dataset, shuffle=True, **loader_params)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_params)

    return train_loader, val_loader

    return train_loader, val_loader