import torch

from ..glow import *
import helper


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

    def prep_conds(self, left_glow_out, extra_cond, direction):
        act_cond = left_glow_out['all_act_outs']
        w_cond = left_glow_out['all_w_outs']
        coupling_cond = left_glow_out['all_flows_outs']

        # Verarbeite numerische Conditions wenn vorhanden
        extra_cond_features = None
        if self.has_extra_cond and extra_cond is not None:
            # Debug print
            print(f"Extra cond shape before extra_cond net: {extra_cond.shape}")
            
            # Ensure extra_cond has correct shape
            if len(extra_cond.shape) == 1:
                extra_cond = extra_cond.unsqueeze(0)  # Add batch dimension
                
            extra_cond_features = self.extra_cond_net(extra_cond)  
            print(f"extra_cond features shape after net: {extra_cond_features.shape}")

        # Für jeden Block und Flow die Bedingungen vorbereiten
        for block_idx in range(len(act_cond)):
            for flow_idx in range(len(act_cond[block_idx])):
                base_features = act_cond[block_idx][flow_idx]
                batch_size, _, height, width = base_features.shape
                
                if extra_cond_features is not None:
                    # Ensure extra_cond_features matches batch size
                    if extra_cond_features.size(0) != batch_size:
                        extra_cond_features = extra_cond_features.expand(batch_size, -1)
                    
                    # Erweitere numerische Features auf räumliche Dimensionen
                    num_features = extra_cond_features.unsqueeze(-1).unsqueeze(-1)
                    num_features = num_features.expand(-1, -1, height, width)
                    
                    print(f"Base features shape: {base_features.shape}")
                    print(f"extra_cond features expanded shape: {num_features.shape}")
                    
                    # Konkateniere Features
                    combined_features = torch.cat([base_features, num_features], dim=1)
                    print(f"Combined features shape: {combined_features.shape}")
                else:
                    combined_features = base_features

                # Update conditions
                act_cond[block_idx][flow_idx] = combined_features
                w_cond[block_idx][flow_idx] = combined_features
                coupling_cond[block_idx][flow_idx] = combined_features

        # Erstelle Conditions Dictionary
        conditions = {
            'act_cond': act_cond,
            'w_cond': w_cond,
            'coupling_cond': coupling_cond
        }

        # Kehre Listen für Reverse-Operation um
        if direction == 'reverse':
            conditions['act_cond'] = [list(reversed(cond)) for cond in list(reversed(conditions['act_cond']))]
            conditions['w_cond'] = [list(reversed(cond)) for cond in list(reversed(conditions['w_cond']))]
            conditions['coupling_cond'] = [list(reversed(cond)) for cond in list(reversed(conditions['coupling_cond']))]
        
        return conditions

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
                    
                    # Concatenate with base features
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

        if direction == 'reverse':
            conditions['act_cond'] = [list(reversed(cond)) for cond in list(reversed(conditions['act_cond']))]
            conditions['w_cond'] = [list(reversed(cond)) for cond in list(reversed(conditions['w_cond']))]
            conditions['coupling_cond'] = [list(reversed(cond)) for cond in list(reversed(conditions['coupling_cond']))]

        return conditions

    def reverse(self, x_a=None, z_b_samples=None, extra_cond=None, reconstruct=False):
        left_glow_out = self.left_glow(x_a)
        
        if self.has_extra_cond and extra_cond is not None:
            # Ensure extra_cond is on same device as model
            if not isinstance(extra_cond, torch.Tensor):
                extra_cond = torch.tensor(extra_cond, dtype=torch.float32)
            extra_cond = extra_cond.to(self.extra_net.net[0].weight.device)
            extra_features = self.extra_net(extra_cond)
            conditions = self.prep_conds(left_glow_out, extra_features, direction='reverse')
        else:
            conditions = self.prep_conds(left_glow_out, None, direction='reverse')
            
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
