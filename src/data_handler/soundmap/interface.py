from . import loader
from globals import device
import helper

def init_soundmap_loaders(args, params):
    dataset_params = {
        'data_folder': params['data_folder'],
        'img_size': params['img_size']
    }

    loader_params = {
        "batch_size": params['batch_size'],
        'num_workers': 0
    }

    loaders = loader.init_loaders(loader_params, dataset_params)
    return loaders

def extract_batches(batch, args):
    if args.dataset == 'soundmap':
        building_batch = batch['building'].to(device)
        soundmap_batch = batch['soundmap'].to(device)
        
        # Überprüfe Dimensionen
        assert building_batch.size(1) == 3, f"Building batch should have 3 channels, got {building_batch.size(1)}"
        assert soundmap_batch.size(1) == 3, f"Soundmap batch should have 3 channels, got {soundmap_batch.size(1)}"
        
        if args.direction == 'building2soundmap':
            left_batch = building_batch
            right_batch = soundmap_batch
        else:
            left_batch = soundmap_batch 
            right_batch = building_batch

        extra_cond_batch = None
        return left_batch, right_batch, extra_cond_batch