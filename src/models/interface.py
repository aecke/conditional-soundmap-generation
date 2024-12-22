from torch import optim
import torch

from .glow import *
from .two_glows import TwoGlows
from .interface_c_glow import *
from globals import real_conds_abs_path
import data_handler
import helper


def take_samples(args, params, model, reverse_cond, n_samples=None):
    with torch.no_grad():
        if 'c_glow' in args.model:
            temp = params['temperature']
            cond = reverse_cond['segment'] if args.direction == 'label2photo' else reverse_cond['real']
            sampled_images, _ = model(x=cond, reverse=True, eps_std=temp)

        elif args.dataset == 'soundmap':
            num_samples = n_samples if n_samples is not None else params['n_samples']
            
            # Stelle sicher dass reverse_cond die richtige Batch-Size hat
            if reverse_cond is not None:
                # Erst Kanäle auf 3
                if reverse_cond.size(1) == 1:
                    reverse_cond = reverse_cond.repeat(1, 3, 1, 1)
                # Dann auf die KORREKTE Batch-Size bringen
                reverse_cond = reverse_cond[:1].repeat(num_samples, 1, 1, 1)
            
            z_samples = sample_z(n_samples=num_samples,
                                temperature=params['temperature'],
                                channels=params['channels'],
                                img_size=params['img_size'],
                                n_block=params['n_block'],
                                split_type=model.split_type)

            if args.direction == 'building2soundmap':
                sampled_images = model.reverse(
                    x_a=reverse_cond,  # Jetzt mit korrekter Batch-Size
                    z_b_samples=z_samples,
                    extra_cond=batch['extra_cond']
                ).cpu().data

        else:
            num_samples = n_samples if n_samples is not None else params['n_samples']
            z_samples = sample_z(n_samples=num_samples,
                               temperature=params['temperature'],
                               channels=params['channels'],
                               img_size=params['img_size'],
                               n_block=params['n_block'],
                               split_type=model.split_type)

            if args.dataset == 'cityscapes' and args.direction == 'label2photo':
                sampled_images = model.reverse(x_a=reverse_cond['segment'],
                                           extra_cond=reverse_cond['boundary'],
                                           z_b_samples=z_samples).cpu().data

            elif args.dataset == 'cityscapes' and args.direction == 'photo2label':
                sampled_images = model.reverse(x_a=reverse_cond['real'],
                                           z_b_samples=z_samples).cpu().data

            elif args.dataset == 'cityscapes' and args.direction == 'bmap2label':
                sampled_images = model.reverse(x_a=reverse_cond['boundary'],
                                           z_b_samples=z_samples).cpu().data

            elif args.dataset == 'maps':
                sampled_images = model.reverse(x_a=reverse_cond,
                                           z_b_samples=z_samples).cpu().data
            else:
                raise NotImplementedError

        return sampled_images


def verify_invertibility(args, params):
    segmentations, real_imgs, boundaries = [torch.rand((1, 3, 256, 256)).to(device)] * 3
    model = init_model(args, params)
    x_a_rec, x_b_rec = model.reconstruct_all(x_a=segmentations, x_b=real_imgs, b_map=boundaries)
    sanity_check(segmentations, real_imgs, x_a_rec, x_b_rec)


def init_model_configs(args):
    assert 'improved' in args.model  # otherwise not implemented yet
    left_configs = {'all_conditional': False, 'split_type': 'regular', 'do_lu': False, 'grad_checkpoint': False}  # default
    right_configs = {'all_conditional': True, 'split_type': 'regular', 'do_lu': False, 'condition': 'left', 'grad_checkpoint': False}  # default condition from left glow

    if 'improved' in args.model:
        if args.do_lu:
            left_configs['do_lu'] = True
            right_configs['do_lu'] = True

        if args.grad_checkpoint:
            left_configs['grad_checkpoint'] = True
            right_configs['grad_checkpoint'] = True

        if args.use_bmaps:
            right_configs['condition'] = 'left + b_maps'

        if args.use_bmaps and args.do_ceil:
            right_configs['condition'] = 'left + b_maps_ceil'

    # Numerische Bedingungen nur hinzufügen wenn das Dataset soundmap ist
    if args.dataset == 'soundmap':
        n_extra_cond = sum([
            getattr(args, 'use_temperature', False),
            getattr(args, 'use_humidity', False),
            getattr(args, 'use_db', False)
        ])
        right_configs['n_extra_cond'] = n_extra_cond
        
        right_configs.update({
            'use_temperature': getattr(args, 'use_temperature', False),
            'use_humidity': getattr(args, 'use_humidity', False),
            'use_db': getattr(args, 'use_db', False)
        })

    print(f'In [init_configs]: configs init done: \nleft_configs: {left_configs} \nright_configs: {right_configs}\n')
    return left_configs, right_configs


def init_model(args, params):
    assert len(params['n_flow']) == params['n_block']

    if args.model == 'c_flow' or 'improved' in args.model:
        left_configs, right_configs = init_model_configs(args)
        model = TwoGlows(params, left_configs, right_configs)

    elif 'c_glow' in args.model:
        model = init_c_glow(args, params)

    else:
        raise NotImplementedError

    print(f'In [init_model]: init model done. Model is on: {device}')
    helper.print_info(args, params, model, which_info='model')
    return model.to(device)


def init_and_load(args, params, run_mode):
    checkpoints_path = helper.compute_paths(args, params)['checkpoints_path']
    optim_step = args.last_optim_step
    model = init_model(args, params)

    if run_mode == 'infer':
        model, _, _, _ = helper.load_checkpoint(checkpoints_path, optim_step, model, None, resume_train=False)
        print(f'In [init_and_load]: returned model for inference')
        return model

    else:  # train
        optimizer = optim.Adam(model.parameters(), lr=params['lr'])
        print(f'In [init_and_load]: returned model and optimizer for training')
        model, optimizer, _, lr = helper.load_checkpoint(checkpoints_path, optim_step, model, optimizer, resume_train=True)
        return model, optimizer, lr

