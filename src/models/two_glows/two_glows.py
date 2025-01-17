from torchvision import transforms
import numpy as np
import torch
from copy import deepcopy
import torch.nn as nn

from globals import device
from models.glow.utils import calc_z_shapes, calc_inp_shapes, calc_cond_shapes, calc_cond_shapes_with_extra # <-- Hier fehlten die imports
from ..glow.cond_net import ExtraCondNet
from ..glow.utils import *
from ..glow import init_glow

#updated extra cond handling

class TwoGlows(nn.Module):
    def __init__(self, params, left_configs, right_configs):
        super().__init__()
        self.left_configs, self.right_configs = left_configs, right_configs
        self.split_type = right_configs['split_type']
        
        input_shapes = calc_inp_shapes(
            params['channels'],
            params['img_size'],
            params['n_block'],
            self.split_type
        )

        # Setup für Extra Conditions
        self.use_temperature = right_configs.get('use_temperature', False)
        self.use_humidity = right_configs.get('use_humidity', False)
        self.use_db = right_configs.get('use_db', False)
        
        n_extra = sum([self.use_temperature, self.use_humidity, self.use_db])
        self.has_extra_cond = n_extra > 0
        
        if self.has_extra_cond:
            self.hidden_dim = 64 
            self.extra_output_dim = 32
            self.extra_net = ExtraCondNet(
                n_features=n_extra,
                hidden_dim=self.hidden_dim,
                output_dim=self.extra_output_dim
            )
            
            condition = right_configs['condition']
            cond_shapes = calc_cond_shapes_with_extra(
                params['channels'],
                params['img_size'],
                params['n_block'],
                self.split_type,
                condition,
                self.extra_output_dim
            )
        else:
            condition = right_configs['condition']
            cond_shapes = calc_cond_shapes(
                params['channels'],
                params['img_size'],
                params['n_block'],
                self.split_type,
                condition
            )

        self.left_glow = init_glow(n_blocks=params['n_block'],
                                    n_flows=params['n_flow'],
                                    input_shapes=input_shapes,
                                    cond_shapes=None,
                                    configs=left_configs)

        self.right_glow = init_glow(n_blocks=params['n_block'],
                                    n_flows=params['n_flow'],
                                    input_shapes=input_shapes,
                                    cond_shapes=cond_shapes,
                                    configs=right_configs)

    def forward(self, x_a, x_b, extra_cond=None):
        """
        Args:
            x_a: Building input
            x_b: Soundmap input
            extra_cond: Extra conditions (temperature, humidity, db)
        """
        left_glow_out = self.left_glow(x_a)

        # Verarbeite extra conditions wenn vorhanden
        if self.has_extra_cond and extra_cond is not None:
            extra_features = self.extra_net(extra_cond)
            conditions = self.prep_conds(left_glow_out, extra_features, direction='forward')
        else:
            conditions = self.prep_conds(left_glow_out, None, direction='forward')

        right_glow_out = self.right_glow(x_b, conditions)

        # Extract and gather outputs
        left_glow_outs = {
            'log_p': left_glow_out['log_p_sum'],
            'log_det': left_glow_out['log_det'],
            'z_outs': left_glow_out['z_outs'],
            'flows_outs': left_glow_out['all_flows_outs']
        }

        right_glow_outs = {
            'log_p': right_glow_out['log_p_sum'],
            'log_det': right_glow_out['log_det'],
            'z_outs': right_glow_out['z_outs'],
            'flows_outs': right_glow_out['all_flows_outs']
        }

        return left_glow_outs, right_glow_outs

    def prep_conds(self, left_glow_out, extra_features, direction):
        act_cond = left_glow_out['all_act_outs']
        w_cond = left_glow_out['all_w_outs']
        coupling_cond = left_glow_out['all_flows_outs']

        # Entferne die unnötige Umkehrung der Listen im Reverse-Fall
        # if direction == 'reverse':
        #     act_cond = [list(reversed(cond)) for cond in list(reversed(act_cond))]
        #     w_cond = [list(reversed(cond)) for cond in list(reversed(w_cond))]
        #     coupling_cond = [list(reversed(cond)) for cond in list(reversed(coupling_cond))]

        for block_idx in range(len(act_cond)):
            for flow_idx in range(len(act_cond[block_idx])):
                base_features = act_cond[block_idx][flow_idx]
                batch_size, _, height, width = base_features.shape

                if extra_features is not None:
                    # Ensure extra_features matches batch size
                    if extra_features.size(0) != batch_size:
                        extra_features = extra_features.expand(batch_size, -1)

                    # Expand extra features to spatial dimensions
                    extra_spatial = extra_features.unsqueeze(-1).unsqueeze(-1)
                    extra_spatial = extra_spatial.expand(-1, -1, height, width)
                    
                    # Korrekte Konkatenation außerhalb der Schleife
                    combined_features = torch.cat([base_features, extra_spatial], dim=1)
                else:
                    combined_features = base_features

                act_cond[block_idx][flow_idx] = combined_features
                w_cond[block_idx][flow_idx] = combined_features
                coupling_cond[block_idx][flow_idx] = combined_features

        conditions = {
            'act_cond': act_cond,
            'w_cond': w_cond,
            'coupling_cond': coupling_cond
        }

        return conditions

    def reverse(self, x_a=None, z_b_samples=None, extra_cond=None, reconstruct=False):
        left_glow_out = self.left_glow(x_a)
        
        # Immer Extra-Conditions verwenden, wenn self.has_extra_cond True ist
        if self.has_extra_cond:
            # Ensure extra_cond is on same device as model
            if not isinstance(extra_cond, torch.Tensor):
                extra_cond = torch.tensor(extra_cond, dtype=torch.float32)
            extra_cond = extra_cond.to(self.extra_net.net[0].weight.device)
            extra_features = self.extra_net(extra_cond)
        else:
            extra_features = None
            
        conditions = self.prep_conds(left_glow_out, extra_features, direction='reverse')

        x_b_syn = self.right_glow.reverse(z_b_samples, reconstruct=reconstruct, conditions=conditions)
        return x_b_syn

    def new_condition(self, x_a, z_b_samples):
        left_glow_out = self.left_glow(x_a)
        conditions = self.prep_conds(left_glow_out, b_map=None, direction='reverse')  # should be tested
        x_b_rec = self.right_glow.reverse(z_b_samples, reconstruct=True, conditions=conditions)
        return x_b_rec

    def reconstruct_all(self, x_a, x_b, b_map=None):
        left_glow_out = self.left_glow(x_a)
        print('left forward done')

        z_outs_left = left_glow_out['z_outs']
        conditions = self.prep_conds(left_glow_out, b_map, direction='forward')  # preparing for right glow forward
        right_glow_out = self.right_glow(x_b, conditions)
        z_outs_right = right_glow_out['z_outs']
        print('right forward done')

        # reverse operations
        x_a_rec = self.left_glow.reverse(z_outs_left, reconstruct=True)
        print('left reverse done')
        
        # need to do forward again since left_glow_out has been changed after preparing condition
        left_glow_out = self.left_glow(x_a)
        conditions = self.prep_conds(left_glow_out, b_map, direction='reverse')  # prepare for right glow reverse
        x_b_rec = self.right_glow.reverse(z_outs_right, reconstruct=True, conditions=conditions)
        print('right reverse done')
        return x_a_rec, x_b_rec


def print_all_shapes(input_shapes, cond_shapes, params, split_type):  # for debugging
    z_shapes = calc_z_shapes(params['channels'], params['img_size'], params['n_block'], split_type)
    # helper.print_and_wait(f'z_shapes: {z_shapes}')
    # helper.print_and_wait(f'input_shapes: {input_shapes}')
    # helper.print_and_wait(f'cond_shapes: {cond_shapes}')
    print(f'z_shapes: {z_shapes}')
    print(f'input_shapes: {input_shapes}')
    print(f'cond_shapes: {cond_shapes}')