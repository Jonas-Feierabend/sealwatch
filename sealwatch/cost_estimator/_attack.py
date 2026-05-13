import conseal as cl 
from conseal.lsb._costmap import Change
import numpy as np 

class costCalculator(): 
    def __init__(self):
        self.cost_functions = [self.hill, self.hugo, self.lsbm, self.lsbr, self.suniward, self.wow]

    def hill(self,x0): 
        return cl.hill.compute_cost_adjusted(x0)

    def hugo(self,x0): 
        return cl.hugo.compute_cost_adjusted(x0)

    def lsbm(self,x0): 
        return cl.lsb.compute_cost_adjusted(x0, modify = Change.LSB_MATCHING)
    def lsbr(self,x0): 
        return cl.lsb.compute_cost_adjusted(x0, modify = Change.LSB_REPLACEMENT)
    def suniward(self,x0): 
        return cl.suniward.compute_cost_adjusted(x0)

    def wow(self,x0): 
        return cl.wow.compute_cost_adjusted(x0)
cc = costCalculator()


import math 
def binary_entropy(p):
    if p <= 0 or p >= 1:
        return 0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

def estimate_parameters(delta, rho_matrix):
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
        p_i = exponent / (1 + 2 * exponent)
        
        beta_theo = np.mean(2 * p_i) # Faktor 2 für +1 und -1
        
        if abs(beta_theo - beta_obs) < 1e-6:
            break
            
        lambda_param *= (beta_theo / beta_obs)
        

    p0 = 1 - 2*p_i
    h_i = - (2 * p_i * np.log2(p_i + 1e-15) + p0 * np.log2(p0 + 1e-15))
    m_estimated = np.sum(h_i)
    
    return lambda_param, m_estimated


def attack(stego, cover): 
        delta = stego.astype(np.int16) - cover.astype(np.int16)
        
        results = []
        for cost_func in cc.cost_functions:
            input_img = cover[:,:,0] if cover.ndim == 3 else cover
            
            # WICHTIG: Ergebnis der Funktion direkt in ein NumPy-Array umwandeln
            rho = np.asarray(cost_func(input_img), dtype=np.float64)
            
            # Auch sicherstellen, dass delta ein Array ist
            delta_arr = np.asarray(delta, dtype=np.float64)
            
            # EM-Schätzung
            est_lambda, est_M = estimate_parameters(delta_arr, rho)
            
            # Hier passiert jetzt kein Fehler mehr, da est_lambda (float) * rho (array) funktioniert
            exponent = np.exp(-est_lambda * rho)
            p_i = exponent / (1 + 2 * exponent)
            p0 = 1 - 2 * p_i
            
            # Log-Likelihood der Beobachtung delta
            # Wir nutzen 1e-15 um log(0) zu vermeiden
            log_lik = np.sum(np.where(delta_arr != 0, np.log(p_i + 1e-15), np.log(p0 + 1e-15)))
            
            results.append({
                'method': cost_func.__name__,
                'M': est_M,
                'lambda': est_lambda,
                'log_lik': log_lik
            })

        # Die Methode mit der höchsten Log-Likelihood ist dein Sieger
        best_res = max(results, key=lambda x: x['log_lik'])
        print(f"Bild : Beste Methode {best_res['method']}, Geschätztes M: {int(best_res['M'])} Bits")
        for e in results: 
            print(e)
        print("\n\n")
        return [best_res['method'], best_res["M"]]