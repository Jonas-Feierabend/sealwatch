import conseal as cl 
from conseal.lsb._costmap import Change
import numpy as np 
import math 
from functools import partial


def binary_entropy(p):
    if p <= 0 or p >= 1:
        return 0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)





def estimate_parameters(delta, rho_matrix, name):
    """
    EM-artiger Ansatz: Finde das optimale Lambda für eine gegebene Kostenmatrix.
    """
    rho_matrix = np.asarray(rho_matrix) 
    delta = np.asarray(delta)
    
    beta_obs = (delta != 0).mean()
    # Startwert für Lambda (Lagrange-Multiplikator)
    lambda_param = 1.0 
    
    for iteration in range(100):

        exponent = np.exp(-lambda_param * rho_matrix)

        if name == "lsbr": 
            p_i = exponent / (1 + 1 * exponent)
            beta_theo = np.mean(1 * p_i) # Faktor 2 für +1 und -1    
        else: 
            p_i = exponent / (1 + 2 * exponent)
            
            beta_theo = np.mean(2 * p_i) # Faktor 2 für +1 und -1
        
        if abs(beta_theo - beta_obs) < 1e-6:
            break
            
        lambda_param *= (beta_theo / beta_obs)
        
    if name == "lsbr": 
        p_i = np.exp(-lambda_param * rho_matrix) / (1 + np.exp(-lambda_param * rho_matrix))
        h_i = -(p_i * np.log2(p_i + 1e-15) + (1 - p_i) * np.log2(1 - p_i + 1e-15))
    else: 
        p_i = np.exp(-lambda_param * rho_matrix) / (1 + np.exp(-lambda_param * rho_matrix))
        p0 = 1 - 2*p_i
        h_i = - (2 * p_i * np.log2(p_i + 1e-15) + p0 * np.log2(p0 + 1e-15))
    m_estimated = np.sum(h_i)
    
    return lambda_param, m_estimated


def attack(stego, cover): 
        delta = stego.astype(np.int16) - cover.astype(np.int16)

        cost_functions = {
            "hill": cl.hill.compute_cost_adjusted,
            "hugo": cl.hugo.compute_cost_adjusted, 
            "suniward": cl.suniward.compute_cost_adjusted,
            "wow": cl.wow.compute_cost_adjusted, 
            "lsbm": partial(cl.lsb.compute_cost_adjusted, modify = Change.LSB_MATCHING), 
            "lsbr": partial(cl.lsb.compute_cost_adjusted, modify = Change.LSB_REPLACEMENT)
        }

        input_img = cover[:,:,0] if cover.ndim == 3 else cover
        delta_2d  = delta[:,:,0] if delta.ndim == 3 else delta 
        delta_arr = np.asarray(delta_2d, dtype=np.float64)
        
        results = []
        for name, cost_func in cost_functions.items():

            rho_raw = np.asarray(cost_func(input_img), dtype=np.float64)  # (2, H, W)

            # (2, H, W) → (H, W) kollabieren
            if name == "lsbr":
                even_mask = (input_img % 2 == 0)
                rho = np.where(even_mask, rho_raw[0], rho_raw[1])
            else:
                rho = rho_raw[0]  # symmetrisch, beide Kanäle gleich
                
            
            est_lambda, est_M = estimate_parameters(delta_arr, rho, name)
            
            
            exponent = np.exp(-est_lambda * rho)
            if name == "lsbr":
                cover_flat = input_img.flatten()
                delta_flat = delta_arr.flatten()
                even_mask = (cover_flat % 2 == 0)
                impossible = np.any(
                    (even_mask & (delta_flat == -1)) | (~even_mask & (delta_flat == +1))
                )
                if impossible:
                    log_lik = -np.inf
                else:
                    p_i = exponent / (1 + exponent)
                    p0  = 1 - p_i
                    log_lik = float(np.sum(np.where(
                        delta_arr != 0,
                        np.log(p_i + 1e-15),
                        np.log(p0  + 1e-15)
                    )))
            else: 
                p_i = exponent / (1 + 2 * exponent)
                p0 = 1 - 2 * p_i
                log_lik = np.sum(np.where(delta_arr != 0, np.log(p_i + 1e-15), np.log(p0 + 1e-15)))
            
            results.append({
                'method_name': name, 
                'M': est_M,
                'lambda': est_lambda,
                'log_lik': log_lik
            })


        best_res = max(results, key=lambda x: x['log_lik'])

        return [[best_res['method_name'], best_res["M"]], results]