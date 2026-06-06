"""
Blind steganalysis attack estimating the embedding method and message length from a cover-stego pair.

Author: Jonas Feierabend 
Affiliation: University of Innsbruck
"""

import conseal as cl
from conseal.lsb import Change
import numpy as np
import math
from functools import partial



def _log_likelihood(delta, exponent, ternary=True):
    """Computes the log-likelihood of observed changes under a given embedding model.

    :param delta: Observed change array (non-zero where a change occurred).
    :type delta: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param exponent: Precomputed exp(-lambda * rho) values.
    :type exponent: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param ternary: If ``True``, uses ternary model. If ``False``, uses binary model.
    :type ternary: bool
    :return: Total log-likelihood.
    :rtype: float
    """

    # ternary models can change in both direction delta in {-1, 0, +1}
    if ternary:
        p_i = exponent / (1 + 2 * exponent) # probability of change
        p0 = 1 - 2 * p_i # probability of no change
    # binary models can only change in one direction delta in {1, 0}
    else:
        p_i = exponent / (1 + exponent)
        p0 = 1 - p_i

    return float(
        np.sum(
            np.where(
                delta != 0,
                np.log(p_i + 1e-15), # changed pixels
                np.log(p0 + 1e-15), # unchanged pixels 
            )
        )
    )

def _log_likelihood_directional(delta, exp_p1, exp_m1):
    """Computes the log-likelihood of observed changes under a directional embedding model.

    :param delta: Observed change array.
    :type delta: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param exp_p1: Precomputed exp(-lambda * rho_p1) values.
    :type exp_p1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param exp_m1: Precomputed exp(-lambda * rho_m1) values.
    :type exp_m1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: Total log-likelihood.
    :rtype: float
    """
    denom = 1 + exp_p1 + exp_m1
    p_plus  = exp_p1 / denom # probability of +1 
    p_minus = exp_m1 / denom # probability of -1 
    p0      = 1 / denom # probability of no change

    return float(np.sum(
        np.where(delta == +1, np.log(p_plus  + 1e-15),
        np.where(delta == -1, np.log(p_minus + 1e-15),
                              np.log(p0      + 1e-15)))
    ))



def _is_impossible(name, cover_dct, delta , wrong_direction, zero_changed):
    """Checks whether an embedding method is structurally impossible given the observed changes.

    Applies three impossibility criteria:

    - **lsb**: LSB-replacement violates parity — even coefficients can only increase,
      odd coefficients can only decrease. Any violation rules out LSB.
    - **nsf5/f5**: nsF5 and F5 always decrease the absolute value of non-zero AC
      coefficients. If any change increases the absolute value, both are ruled out.
    - **f5**: F5 never modifies zero coefficients. If any zero coefficient changed,
      F5 is ruled out.

    :param name: Name of the embedding method to check.
    :type name: str
    :param cover_dct: Cover DCT coefficients as 4D array.
    :type cover_dct: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param delta: Difference array between stego and cover DCT coefficients.
    :type delta: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param wrong_direction: ``True`` if any change increases the absolute value of a non-zero AC coefficient.
    :type wrong_direction: bool
    :param zero_changed: ``True`` if any zero coefficient was modified.
    :type zero_changed: bool
    :return: ``True`` if the method is structurally impossible, ``False`` otherwise.
    :rtype: bool
    """

    # lsb is the lsbr äquivalent in the jpeg domain 
    if name in ("lsb", "lsbr"):
        # checks if lsb makes any unallowed changes 
        cover_flat = cover_dct.ravel()
        delta_flat = delta.ravel()
        even_mask = cover_flat % 2 == 0
        return bool(np.any(
            (even_mask & (delta_flat == -1)) | (~even_mask & (delta_flat == +1))
        ))
    if name in ("nsf5", "f5") and wrong_direction:
        # both cannot increase values 
        return True
    if name in ("nsf5", "f5") and zero_changed:
        # both cannot change zero coefficients 
        return True
    return False


def estimate_lambda(delta, rho_matrix, ternary=True, max_iter=200, damping_factor=0.5):
    """Estimates the optimal Lagrange multiplier for a given cost matrix.

    Uses a multiplicative update to find lambda such that the theoretical
    change rate matches the observed change rate.

    :param delta: Difference array between stego and cover.
    :type delta: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param rho_matrix: Cost matrix for the embedding method.
    :type rho_matrix: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param ternary: If ``True``, uses ternary embedding model (delta ∈ {-1, 0, +1}).
        If ``False``, uses binary model (delta ∈ {0, 1}).
    :type ternary: bool
    :param max_iter: Maximum number of iterations for the multiplicative update.
    :type max_iter: int
    :param damping_factor: Exponent for the multiplicative update step (< 1 slows convergence for stability).
    :type damping_factor: float
    :return: Estimated Lagrange multiplier lambda.
    :rtype: float

    :Example:

    >>> import numpy as np
    >>> lam = estimate_lambda(np.array([1, 0, 1]), np.ones(3), ternary=False)
    """
    rho_matrix = np.asarray(rho_matrix)
    delta = np.asarray(delta)

    beta_obs = (delta != 0).mean()
    lambda_param = 1.0
    if beta_obs == 0:
        return 0.0  # no changes observed -> no embedding
    
    for _ in range(max_iter):
        exponent = np.exp(-lambda_param * rho_matrix)

        # calculate Gibbs distribution 
        if ternary: # delta in [-1,0,+1]
            p_i = exponent / (1 + 2 * exponent)
            beta_theo = np.mean(2 * p_i)
        else: # delta in [0, 1]
            p_i = exponent / (1 + exponent)
            beta_theo = np.mean(p_i)

        # early exit if result is good enough 
        if abs(beta_theo - beta_obs) < 1e-10:
            break

        # damping of 0.5 results in good convergence 
        lambda_param *= (beta_theo / beta_obs)**damping_factor

    return lambda_param

def estimate_lambda_directional(delta, rho_p1, rho_m1, max_iter=200, damping_factor=0.5):
    """Estimates lambda using directional cost matrices.

    Uses a multiplicative update to find lambda such that the theoretical
    change rate matches the observed change rate, accounting for asymmetric
    embedding costs in the plus and minus directions.

    :param delta: Difference array between stego and cover.
    :type delta: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param rho_p1: Cost matrix for +1 changes.
    :type rho_p1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param rho_m1: Cost matrix for -1 changes.
    :type rho_m1: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param max_iter: Maximum number of iterations for the multiplicative update.
    :type max_iter: int
    :param damping_factor: Exponent for the multiplicative update step (< 1 slows convergence for stability).
    :type damping_factor: float
    :return: Estimated Lagrange multiplier lambda.
    :rtype: float

    :Example:

    >>> import numpy as np
    >>> lam = estimate_lambda_directional(np.array([1, 0, -1]), np.ones(3), np.ones(3))
    """
    delta = np.asarray(delta)
    beta_obs = (delta != 0).mean()

    if beta_obs == 0:
        # No changes observed 
        return 0.0

    lambda_param = 1.0
    for _ in range(max_iter):
        # calculate Gibbs distribution for directional rho
        exp_p1 = np.exp(-lambda_param * rho_p1)
        exp_m1 = np.exp(-lambda_param * rho_m1)
        denom = 1 + exp_p1 + exp_m1
        beta_theo = np.mean((exp_p1 + exp_m1) / denom)

        # early exit if result is good enough 
        if abs(beta_theo - beta_obs) < 1e-10:
            break

        # damping of 0.5 results in good convergence 
        lambda_param *= (beta_theo / beta_obs)**damping_factor

    return lambda_param




def attack(stego, cover, cover_spatial=None, qtable=None):
    """Identify the embedding method and estimate the message length from a cover-stego pair.

    Auto-detects the domain based on input shape: 4D arrays are treated as
    DCT coefficients (JPEG domain), 2D/3D arrays as spatial domain images.

    :param stego: Stego image or DCT coefficients.
    :type stego: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param cover: Cover image or DCT coefficients, same shape as stego.
    :type cover: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param cover_spatial: Cover image in spatial domain. Required for JPEG attack.
    :type cover_spatial: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__, optional
    :param qtable: JPEG quantization table. Required for JPEG attack.
    :type qtable: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__, optional
    :return: Dictionary with keys ``'method'`` (str), ``'M'`` (float, estimated message
        length in bits), ``'lambda'`` (float, Lagrange multiplier), and ``'all'``
        (list of results for each candidate method).
    :rtype: dict
    :raises AssertionError: If JPEG domain is detected but ``cover_spatial`` or
        ``qtable`` are not provided.
    :raises ValueError: If stego and cover are identical.
    """

    if stego.ndim == 4:
        # 4D -> DCT-coefficient (JPEG)
        assert cover_spatial is not None and qtable is not None, \
            "cover_spatial and qtable required for JPEG attack"
        return attack_jpeg(stego, cover, cover_spatial, qtable)
    else:
        # 2D/3D -> Spatial-Domain
        return attack_spatial(stego, cover)

def attack_spatial(stego, cover):
    """Runs a blind steganalysis attack on spatial domain images.

    Estimates the most likely embedding method by computing cost matrices
    for all supported methods and selecting the one with the highest
    log-likelihood given the observed changes.

    Supported methods: ``'hill'``, ``'hugo'``, ``'suniward'``, ``'wow'``,
    ``'lsbm'``, ``'lsbr'``.

    :param stego: Stego image as 2D or 3D numpy array.
    :type stego: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param cover: Cover image as 2D or 3D numpy array, same shape as stego.
    :type cover: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: Dictionary with keys ``'method'``, ``'M'``, ``'lambda'``, ``'all'``.
    :rtype: dict

    :Example:

    >>> import numpy as np
    >>> cover = np.array(Image.open("cover.png").convert("L"))
    >>> result = attack_spatial(stego, cover)
    >>> result["method"]
    'hill'
    """

    # predefined cost functions from conseal package 
    cost_functions = {
        "hill": cl.hill.compute_cost_adjusted,
        "hugo": cl.hugo.compute_cost_adjusted,
        "suniward": cl.suniward.compute_cost_adjusted,
        "wow": cl.wow.compute_cost_adjusted,
        "lsbm": partial(
            cl.lsb.compute_cost_adjusted, modify=Change.LSB_MATCHING
        ),
        "lsbr": partial(
            cl.lsb.compute_cost_adjusted, modify=Change.LSB_REPLACEMENT
        ),
    }

    if np.array_equal(stego, cover):
        raise ValueError("stego and cover are identical — did you pass the same image twice?")
    
    delta = stego.astype(np.int16) - cover.astype(np.int16)
    # only take first channel if there are multiple 
    input_img = cover[:, :, 0] if cover.ndim == 3 else cover
    delta_2d = delta[:, :, 0] if delta.ndim == 3 else delta
    delta_arr = np.asarray(delta_2d, dtype=np.float64)

    results = []
    for name, cost_func in cost_functions.items():

        rho_raw = np.asarray(cost_func(input_img), dtype=np.float64)  # (2, H, W)

        if name == "lsbr":
            # transform two rho matrices to one combined
            even_mask = input_img % 2 == 0
            rho = np.where(even_mask, rho_raw[0], rho_raw[1])

            est_lambda = estimate_lambda(delta_arr, rho, ternary=False)
            
            # calculate estimated message length 
            # two times the observed changes is a very good estimate 
            # because 50% of time the pixel is already in the desired state 
            est_M = 2.0 * float((delta_arr != 0).sum())

            # check for not allowed changes 
            impossible = _is_impossible(name, cover, delta, False, False)


            exponent = np.exp(-est_lambda * rho)
            log_lik = -np.inf if impossible else _log_likelihood(delta_arr, exponent, ternary=False)

        elif name == "lsbm":
            # rho is symetric 
            rho = rho_raw[0]
            est_lambda = estimate_lambda(delta_arr, rho, ternary=True)
            

            # calculate estimated message length 
            # two times the observed changes is a very good estimate 
            # because 50% of time the pixel is already in the desired state 
            est_M = 2.0 * float((delta_arr != 0).sum())

            exponent = np.exp(-est_lambda * rho)
            log_lik = _log_likelihood(delta_arr, exponent, ternary=True)

        else:  # hill, hugo, suniward, wow — directional
            # not symetric -> directional 
            rho_p1 = rho_raw[0]
            rho_m1 = rho_raw[1]
            est_lambda = estimate_lambda_directional(delta_arr, rho_p1, rho_m1)

            # calculate message length 
            exp_p1 = np.exp(-est_lambda * rho_p1)
            exp_m1 = np.exp(-est_lambda * rho_m1)
            denom = 1 + exp_p1 + exp_m1
            p0 = 1 / denom
            est_M = float(np.sum(-(
                exp_p1/denom * np.log2(exp_p1/denom + 1e-15) +
                exp_m1/denom * np.log2(exp_m1/denom + 1e-15) +
                p0 * np.log2(p0 + 1e-15)
            )))

            # log likelihood 
            log_lik = _log_likelihood_directional(delta_arr, exp_p1, exp_m1)

        results.append({"method_name": name, "M": est_M, "lambda": est_lambda, "log_lik": log_lik})

    # sort for the highest log likelihood 
    best_res = max(results, key=lambda x: x["log_lik"])

    return {
        "method": best_res["method_name"],
        "M": best_res["M"],
        "lambda": best_res["lambda"],
        "all": results,
    }

def attack_jpeg(stego_dct, cover_dct, cover_spatial, qtable):
    """Runs a blind steganalysis attack on JPEG domain images.

    Estimates the most likely embedding method by computing cost matrices
    for all supported methods and selecting the one with the highest
    log-likelihood given the observed DCT coefficient changes.

    Supported methods: ``'lsb'``, ``'nsf5'``, ``'f5'``, ``'juniward'``,
    ``'uerd'``, ``'ebs'``.

    :param stego_dct: Stego DCT coefficients as 4D numpy array (blocks x blocks x 8 x 8).
    :type stego_dct: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param cover_dct: Cover DCT coefficients, same shape as stego_dct.
    :type cover_dct: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param cover_spatial: Cover image in spatial domain, used for juniward cost computation.
    :type cover_spatial: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param qtable: JPEG quantization table.
    :type qtable: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :return: Dictionary with keys ``'method'``, ``'M'``, ``'lambda'``, ``'all'``.
    :rtype: dict

    :Example:

    >>> import jpeglib
    >>> jpeg = jpeglib.read_dct("image.jpg")
    >>> result = attack_jpeg(stego_dct, jpeg.Y, spatial, jpeg.qt[0])
    >>> result["method"]
    'juniward'
    """

    # predefined methods 
    # "name": (isTernary, function)
    all_methods = {
        # lsb, nsf5, f5: DCT-domain cost functions produce poor results for these methods.
        # Fall back to uniform cost (rho=1) with infinite cost for DC and zero coefficients.
        "lsb":      (False, None),  
        "nsf5":     (False, None), 
        "f5":       (False, None), 
        "juniward": (True, lambda: cl.juniward.compute_cost_adjusted(x0=cover_spatial, y0=cover_dct, qt=qtable)),
        "uerd":     (True, lambda: cl.uerd.compute_cost_adjusted(y0=cover_dct, qt=qtable)),
        "ebs":      (True, lambda: cl.ebs.compute_cost_adjusted(y0=cover_dct, qt=qtable)),
    }

    if np.array_equal(stego_dct, cover_dct):
        raise ValueError("stego_dct and cover_dct are identical — did you pass the same image twice?")
    
    delta = (stego_dct.astype(np.int32) - cover_dct.astype(np.int32)).astype(np.float64)


    # Extra checks for better accuracy 
    embed_mask_f5 = cover_dct != 0
    embed_mask_f5[:, :, 0, 0] = False
    # check if zero coefficients were changed 
    zero_changed = bool(np.any((cover_dct == 0) & (delta != 0)))
    # check if lsbr unconform changes have been made 
    wrong_direction = bool(np.any(
        ((cover_dct > 0) & embed_mask_f5 & (delta == +1))
        | ((cover_dct < 0) & embed_mask_f5 & (delta == -1))
    ))



    results = []
    for name, (ternary, cost_fn) in all_methods.items():

        # extra checks for better accuracy 
        if _is_impossible(name, cover_dct, delta, wrong_direction, zero_changed):
            results.append({"method_name": name, "M": 0.0, "lambda": 0.0, "log_lik": -np.inf})
            continue

        if cost_fn is None:  # uniform cost (lsb, nsf5, f5)
            # create own cost 
            rho = np.ones_like(cover_dct, dtype=np.float64)
            rho[cover_dct == 0] = 1e13
            rho[:, :, 0, 0] = 1e13
        else:
            # rho is symetric -> if two rhos are supplied reduce to one 
            rho_raw = np.asarray(cost_fn(), dtype=np.float64)
            rho = rho_raw[0] if rho_raw.ndim == 5 else rho_raw

        # using the mask for every method because if not 
        # ternary embeddings have an advantage
        delta_fit = delta[embed_mask_f5]
        rho_fit = rho[embed_mask_f5]

        est_lambda = estimate_lambda(delta_fit, rho_fit, ternary=ternary)
        exponent = np.exp(-est_lambda * rho_fit)

        # calculate M 
        if ternary:
            p_i = exponent / (1 + 2 * exponent)
            p0 = 1 - 2 * p_i
            est_M = float(np.sum(-(2 * p_i * np.log2(p_i + 1e-15) + p0 * np.log2(p0 + 1e-15))))
        else:
            est_M = 2.0 * float((delta_fit != 0).sum())

        # calculate log likelihood 
        log_lik = _log_likelihood(delta_fit, exponent, ternary=ternary)
        results.append({"method_name": name, "M": est_M, "lambda": est_lambda, "log_lik": log_lik})

    # sort for highest log likelihood 
    best_res = max(results, key=lambda x: x["log_lik"])
    return {
        "method": best_res["method_name"],
        "M": best_res["M"],
        "lambda": best_res["lambda"],
        "all": results,
    }