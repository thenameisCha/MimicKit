import torch

import envs.base_env as base_env

def compute_td_lambda_return(r, next_vals, done, discount, td_lambda):
    assert(r.shape == next_vals.shape)

    return_t = torch.zeros_like(r)
    reset_mask = done != base_env.DoneFlags.NULL.value
    reset_mask = reset_mask.type(torch.float)

    last_val = r[-1] + discount * next_vals[-1]
    return_t[-1] = last_val

    timesteps = r.shape[0]
    for i in reversed(range(0, timesteps - 1)):
        curr_r = r[i]
        curr_reset = reset_mask[i]
        next_v = next_vals[i]
        next_ret = return_t[i + 1]

        curr_lambda = td_lambda * (1.0 - curr_reset)
        curr_val = curr_r + discount * ((1.0 - curr_lambda) * next_v + curr_lambda * next_ret)
        return_t[i] = curr_val

    #_debug_td_lambda(r, next_vals, done, discount, td_lambda, return_t)

    return return_t

def compute_td_lambda_return_custom(r, next_vals, done, discount, td_lambda, CaT_delta = 0):
    assert(r.shape == next_vals.shape)
        
    # TODO
    # compute CaT pmax with all the CaT_c recored in exp_buffer
    # scale r with CaT values
    r *= (1.0 - CaT_delta)
    return_t = torch.zeros_like(r)
    reset_mask = done != base_env.DoneFlags.NULL.value
    reset_mask = reset_mask.type(torch.float)

    last_val = r[-1] + discount * next_vals[-1]
    return_t[-1] = last_val

    timesteps = r.shape[0]
    for i in reversed(range(0, timesteps - 1)):
        curr_r = r[i]
        curr_reset = reset_mask[i]
        next_v = next_vals[i]
        next_ret = return_t[i + 1]

        curr_lambda = td_lambda * (1.0 - curr_reset) 
        curr_val = curr_r + discount * ((1.0 - curr_lambda) * next_v + curr_lambda * next_ret * (1.0 - CaT_delta[i]))
        return_t[i] = curr_val

    #_debug_td_lambda(r, next_vals, done, discount, td_lambda, return_t)

    return return_t