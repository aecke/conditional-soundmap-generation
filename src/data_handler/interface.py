import os
from PIL import Image  # Änderung: tkinter.Image -> PIL.Image
from torchvision import transforms  # Änderung: matplotlib.transforms -> torchvision.transforms
import torch
from data_handler.soundmap.interface import init_soundmap_loaders
from . import city, maps, soundmap, transient
from globals import maps_fixed_conds, device
from globals import soundmap_test_buildings


def retrieve_rev_cond(args, params, run_mode='train'):
    if args.dataset == 'cityscapes':
        reverse_cond = None if args.exp else city.prepare_city_reverse_cond(args, params, run_mode)
    elif args.dataset == 'maps':
        reverse_cond = maps.create_rev_cond(args, params, fixed_conds=maps_fixed_conds, also_save=True)
    elif args.dataset == 'soundmap':
        if run_mode == 'train':
            # Lade die festgelegten Test-Gebäude
            test_buildings = []
            test_folder = params['data_folder']['test']['buildings']
            transform = transforms.Compose([  # Hier als Variable definiert
                transforms.Resize(params['img_size']),
                transforms.Grayscale(num_output_channels=1),
                transforms.ToTensor(),
            ])
            
            for building_name in soundmap_test_buildings:
                path = os.path.join(test_folder, building_name)
                if os.path.exists(path):
                    img = transform(Image.open(path))
                    test_buildings.append(img)
                else:
                    print(f"Warning: Building file not found: {path}")
                    continue
            
            if not test_buildings:
                raise RuntimeError("No test buildings could be loaded!")
                
            test_buildings = torch.stack(test_buildings).to(device)
            return test_buildings
    else:
        raise NotImplementedError('Dataset not implemented')
    return reverse_cond


def extract_batches(batch, args):
    """
    This function depends onf the dataset and direction.
    :param batch:
    :param args:
    :return:
    """
    if args.dataset == 'cityscapes':
        img_batch = batch['real'].to(device)
        segment_batch = batch['segment'].to(device)
        boundary_batch = batch['boundary'].to(device)

        if args.direction == 'label2photo':
            left_batch = segment_batch
            right_batch = img_batch
            extra_cond_batch = boundary_batch if args.use_bmaps else None

        elif args.direction == 'photo2label':
            left_batch = img_batch
            right_batch = segment_batch
            extra_cond_batch = None

        elif args.direction == 'bmap2label':
            left_batch = boundary_batch
            right_batch = segment_batch
            extra_cond_batch = None

        else:
            raise NotImplementedError

    elif args.dataset == 'maps':
        photo_batch = batch['photo'].to(device)
        map_batch = batch['the_map'].to(device)
        extra_cond_batch = None

        if args.dataset == 'map2photo':
            left_batch = map_batch
            right_batch = photo_batch
        else:
            left_batch = photo_batch
            right_batch = map_batch

    elif args.dataset == 'soundmap':
        building_batch = batch['building'].to(device)
        soundmap_batch = batch['soundmap'].to(device)
        extra_cond = None
    
    # Numerische Bedingungen nur extrahieren wenn die Flags gesetzt sind
    if any([args.use_temperature, args.use_humidity, args.use_db]):
        if 'extra_cond' not in batch:
            print("Warning: Numerical conditions flags are set but no numerical data found in batch")
        else:
            extra_cond = batch['extra_cond'].to(device)
    
    if args.direction == 'building2soundmap':
        left_batch = building_batch
        right_batch = soundmap_batch
    else:
        left_batch = soundmap_batch
        right_batch = building_batch

    return left_batch, right_batch, extra_cond


def init_data_loaders(args, params):
    batch_size = params['batch_size']
    if args.dataset == 'cityscapes':
        loader_params = {'batch_size': batch_size, 'shuffle': True, 'num_workers': 0}
        train_loader, \
            val_loader = city.init_city_loader(data_folder=params['data_folder'],
                                               image_size=(params['img_size']),
                                               loader_params=loader_params,
                                               limited=args.limited)
    elif args.dataset == 'maps':
        train_loader, val_loader = maps.init_maps_loaders(args, params)
        
    elif args.dataset == 'soundmap':
        loader_params = {'batch_size': batch_size, 'num_workers': 0}
        # Train loader
        train_dataset = soundmap.SoundMapDataset(
            data_folder=params['data_folder'],
            is_train=True,
            img_size=params['img_size'],
            use_temperature=args.use_temperature,  # Diese Parameter hinzufügen
            use_humidity=args.use_humidity,
            use_db=args.use_db
        )
        train_loader = torch.utils.data.DataLoader(
            train_dataset, 
            shuffle=True,
            **loader_params
        )

        # Test/Val loader
        val_dataset = soundmap.SoundMapDataset(
            data_folder=params['data_folder'],
            is_train=False,
            img_size=params['img_size'],
            use_temperature=args.use_temperature,  # Diese Parameter hinzufügen
            use_humidity=args.use_humidity,
            use_db=args.use_db
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset, 
            shuffle=False,
            **loader_params
        )
    else:
        raise NotImplementedError

    print(f'\nIn [init_data_loaders]: training with data loaders of size: \n'
          f'train_loader: {len(train_loader):,} \n'
          f'val_loader: {len(val_loader):,} \n'
          f'and batch_size of: {batch_size}\n')
    return train_loader, val_loader