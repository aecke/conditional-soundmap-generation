import torch

from ..glow import *
import helper


class TwoGlows(nn.Module):
    def __init__(self, params, left_configs, right_configs):
        super().__init__()
        self.left_configs, self.right_configs = left_configs, right_configs

        self.split_type = right_configs['split_type']  # this attribute will also be used in take sample
        condition = right_configs['condition']
        input_shapes = calc_inp_shapes(params['channels'],
                                       params['img_size'],
                                       params['n_block'],
                                       self.split_type)

        cond_shapes = calc_cond_shapes(params['channels'],
                                       params['img_size'],
                                       params['n_block'],
                                       self.split_type,
                                       condition)  # shape (C, H, W)

        # print_all_shapes(input_shapes, cond_shapes, params, split_type)

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
        
        self.use_temperature = right_configs.get('use_temperature', False)
        self.use_humidity = right_configs.get('use_humidity', False)
        
        n_numerical = sum([self.use_temperature, self.use_humidity])
        if n_numerical > 0:
            self.numerical_net = NumericalCondNet(n_numerical)

    def prep_conds(self, left_glow_out, numerical_conditions, direction):
        act_cond = left_glow_out['all_act_outs']
        w_cond = left_glow_out['all_w_outs']
        coupling_cond = left_glow_out['all_flows_outs']

        # Für jeden Block und Flow die Bedingungen vorbereiten
        for block_idx in range(len(act_cond)):
            for flow_idx in range(len(act_cond[block_idx])):
                cond_h, cond_w = act_cond[block_idx][flow_idx].shape[2:]
                combined_features = [act_cond[block_idx][flow_idx]]  # Start mit Bild-Features
                
                # 1. Building Features sind immer dabei
                # sind bereits in act_cond enthalten
                
                # 2. Numerische Bedingungen hinzufügen falls vorhanden
                if hasattr(self, 'numerical_net') and isinstance(numerical_conditions, torch.Tensor) and len(numerical_conditions.shape) == 2:
                    num_features = numerical_conditions.unsqueeze(-1).unsqueeze(-1)
                    num_features = num_features.expand(-1, -1, cond_h, cond_w)
                    combined_features.append(num_features)

                # Alle Features zusammenführen
                if len(combined_features) > 1:
                    act_cond[block_idx][flow_idx] = torch.cat(combined_features, dim=1)
                    w_cond[block_idx][flow_idx] = torch.cat(combined_features, dim=1)
                    coupling_cond[block_idx][flow_idx] = torch.cat(combined_features, dim=1)

        # make conds a dictionary
        conditions = make_cond_dict(act_cond, w_cond, coupling_cond)

        # reverse lists for reverse operation
        if direction == 'reverse':
            conditions['act_cond'] = [list(reversed(cond)) for cond in list(reversed(conditions['act_cond']))]
            conditions['w_cond'] = [list(reversed(cond)) for cond in list(reversed(conditions['w_cond']))]
            conditions['coupling_cond'] = [list(reversed(cond)) for cond in list(reversed(conditions['coupling_cond']))]
        
        return conditions

    def forward(self, x_a, x_b, numerical_conditions=None):  # x_a: building, numerical_conditions: numerical conditions
        # perform left glow forward
        left_glow_out = self.left_glow(x_a)

        # Verarbeite numerische Bedingungen wenn vorhanden
        if hasattr(self, 'numerical_net') and numerical_conditions is not None:
            numerical_features = self.numerical_net(numerical_conditions)
            conditions = self.prep_conds(left_glow_out, numerical_features, direction='forward')
        else:
            conditions = self.prep_conds(left_glow_out, numerical_conditions, direction='forward')

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

    def reverse(self, x_a=None, z_b_samples=None, numerical_conditions=None, reconstruct=False, numerical_conditions=None):
        print(f"TwoGlows reverse input shapes: x_a={x_a.shape if x_a is not None else None}, z_b_samples={z_b_samples[0].shape if z_b_samples else None}")
        left_glow_out = self.left_glow(x_a)  # left glow forward always needed before preparing conditions
        conditions = self.prep_conds(left_glow_out, numerical_conditions, direction='reverse')
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
