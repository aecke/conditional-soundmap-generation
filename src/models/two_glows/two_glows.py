import torch

from ..glow import *
import helper


class TwoGlows(nn.Module):
    def __init__(self, params, left_configs, right_configs):
        super().__init__()
        self.left_configs, self.right_configs = left_configs, right_configs
        self.split_type = right_configs['split_type']
        
        # Berechne Base-Input-Shapes ohne numerische Conditions
        input_shapes = calc_inp_shapes(
            params['channels'],
            params['img_size'],
            params['n_block'],
            self.split_type
        )
        
        # Setup für numerische Conditions
        self.use_temperature = right_configs.get('use_temperature', False)
        self.use_humidity = right_configs.get('use_humidity', False)
        self.use_db = right_configs.get('use_db', False)
        
        n_numerical = sum([self.use_temperature, self.use_humidity, self.use_db])
        self.has_numerical = n_numerical > 0
        
        if self.has_numerical:
            self.numerical_hidden_dim = 64
            self.numerical_output_dim = 32  # Reduzierte Feature-Dimension für numerische Daten
            self.numerical_net = NumericalCondNet(
                n_features=n_numerical,
                hidden_dim=self.numerical_hidden_dim,
                output_dim=self.numerical_output_dim
            )
            
            # Aktualisiere Input-Shapes für die Conditions mit zusätzlichen numerischen Features
            condition = right_configs['condition']
            cond_shapes = calc_cond_shapes_with_numerical(
                params['channels'],
                params['img_size'],
                params['n_block'],
                self.split_type,
                condition,
                self.numerical_output_dim
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

        self.left_glow = init_glow(
            n_blocks=params['n_block'],
            n_flows=params['n_flow'],
            input_shapes=input_shapes,
            cond_shapes=None,
            configs=left_configs
        )

        self.right_glow = init_glow(
            n_blocks=params['n_block'],
            n_flows=params['n_flow'],
            input_shapes=input_shapes,
            cond_shapes=cond_shapes,
            configs=right_configs
        )

    def prep_conds(self, left_glow_out, extra_cond, direction):
        act_cond = left_glow_out['all_act_outs']
        w_cond = left_glow_out['all_w_outs']
        coupling_cond = left_glow_out['all_flows_outs']

        # Verarbeite numerische Conditions wenn vorhanden
        numerical_features = None
        if self.has_numerical and extra_cond is not None:
            numerical_features = self.numerical_net(extra_cond)  # Shape: (batch_size, numerical_output_dim)

        # Für jeden Block und Flow die Bedingungen vorbereiten
        for block_idx in range(len(act_cond)):
            for flow_idx in range(len(act_cond[block_idx])):
                base_features = act_cond[block_idx][flow_idx]
                batch_size, _, height, width = base_features.shape
                
                if numerical_features is not None:
                    # Erweitere numerische Features auf räumliche Dimensionen
                    num_features = numerical_features.unsqueeze(-1).unsqueeze(-1)
                    num_features = num_features.expand(-1, -1, height, width)
                    
                    # Konkateniere Features
                    combined_features = torch.cat([base_features, num_features], dim=1)
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

    def forward(self, x_a, x_b, extra_cond=None):  # x_a: building, extra_cond: numerical conditions
        # perform left glow forward
        left_glow_out = self.left_glow(x_a)

        # Verarbeite numerische Bedingungen wenn vorhanden
        if hasattr(self, 'numerical_net') and extra_cond is not None:
            numerical_features = self.numerical_net(extra_cond)
            conditions = self.prep_conds(left_glow_out, numerical_features, direction='forward')
        else:
            conditions = self.prep_conds(left_glow_out, extra_cond, direction='forward')

        # perform right glow forward
        right_glow_out = self.right_glow(x_b, conditions)

        # Rest bleibt unverändert
        # extract left outputs
        log_p_sum_left, log_det_left = left_glow_out['log_p_sum'], left_glow_out['log_det']
        z_outs_left, flows_outs_left = left_glow_out['z_outs'], left_glow_out['all_flows_outs']

        # extract right outputs
        log_p_sum_right, log_det_right = right_glow_out['log_p_sum'], right_glow_out['log_det']
        z_outs_right, flows_outs_right = right_glow_out['z_outs'], right_glow_out['all_flows_outs']

        # gather left outputs together
        left_glow_outs = {'log_p': log_p_sum_left, 'log_det': log_det_left,
                          'z_outs': z_outs_left, 'flows_outs': flows_outs_left}

        #  gather right outputs together
        right_glow_outs = {'log_p': log_p_sum_right, 'log_det': log_det_right,
                           'z_outs': z_outs_right, 'flows_outs': flows_outs_right}

        return left_glow_outs, right_glow_outs

    def reverse(self, x_a=None, z_b_samples=None, extra_cond=None, reconstruct=False):
        print(f"TwoGlows reverse input shapes: x_a={x_a.shape if x_a is not None else None}, z_b_samples={z_b_samples[0].shape if z_b_samples else None}")
        left_glow_out = self.left_glow(x_a)  # left glow forward always needed before preparing conditions
        conditions = self.prep_conds(left_glow_out, extra_cond, direction='reverse')
        x_b_syn = self.right_glow.reverse(z_b_samples, reconstruct=reconstruct, conditions=conditions)  # sample x_b conditioned on x_a
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
