import conseal as cl
from conseal.lsb._costmap import Change
import numpy as np
import math
from functools import partial


def binary_entropy(p):
    """Computes the binary entropy of a probability value.

    Returns 0 for values outside the valid range (0, 1).

    :param p: Probability value.
    :type p: float
    :return: Binary entropy H(p) = -p*log2(p) - (1-p)*log2(1-p)
    :rtype: float

    :Example:

    >>> binary_entropy(0.5)
    1.0
    >>> binary_entropy(0.0)
    0.0
    """
    if p <= 0 or p >= 1:
        return 0
    return -p * math.log2(p) - (1 - p) * math.log2(1 - p)

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
    if ternary:
        p_i = exponent / (1 + 2 * exponent)
        p0 = 1 - 2 * p_i
    else:
        p_i = exponent / (1 + exponent)
        p0 = 1 - p_i

    return float(
        np.sum(
            np.where(
                delta != 0,
                np.log(p_i + 1e-15),
                np.log(p0 + 1e-15),
            )
        )
    )

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
    :param embed_mask_f5: Boolean mask selecting non-zero AC coefficients (DC excluded).
    :type embed_mask_f5: `np.ndarray <https://numpy.org/doc/stable/reference/generated/numpy.ndarray.html>`__
    :param wrong_direction: ``True`` if any change increases the absolute value of a non-zero AC coefficient.
    :type wrong_direction: bool
    :param zero_changed: ``True`` if any zero coefficient was modified.
    :type zero_changed: bool
    :return: ``True`` if the method is structurally impossible, ``False`` otherwise.
    :rtype: bool
    """
    if name == "lsb":
        cover_flat = cover_dct.ravel()
        delta_flat = delta.ravel()
        even_mask = cover_flat % 2 == 0
        return bool(np.any(
            (even_mask & (delta_flat == -1)) | (~even_mask & (delta_flat == +1))
        ))
    if name in ("nsf5", "f5") and wrong_direction:
        return True
    if name == "f5" and zero_changed:
        return True
    return False

def estimate_parameters(delta, rho_matrix, ternary=True):
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
    :return: Estimated Lagrange multiplier lambda.
    :rtype: float

    :Example:

    >>> import numpy as np
    >>> lam = estimate_parameters(np.array([1, 0, 1]), np.ones(3), ternary=False)
    """
    rho_matrix = np.asarray(rho_matrix)
    delta = np.asarray(delta)

    beta_obs = (delta != 0).mean()
    lambda_param = 1.0

    for _ in range(100):
        exponent = np.exp(-lambda_param * rho_matrix)
        if ternary:
            p_i = exponent / (1 + 2 * exponent)
            beta_theo = np.mean(2 * p_i)
        else:
            p_i = exponent / (1 + exponent)
            beta_theo = np.mean(p_i)

        if abs(beta_theo - beta_obs) < 1e-6:
            break

        lambda_param *= beta_theo / beta_obs

    return lambda_param


def attack(stego, cover, cover_spatial=None, qtable=None):
    """Auto-detects spatial vs JPEG domain and calls the appropriate attack."""
    if stego.ndim == 4:
        # 4D -> DCT-coefficient (JPEG)
        assert cover_spatial is not None and qtable is not None, \
            "cover_spatial and qtable required for JPEG attack"
        return attack_jpeg(stego, cover, cover_spatial, qtable)
    else:
        # 2D/3D → Spatial-Domain
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
    >>> result = attack(stego, cover)
    >>> result["method"]
    'hill'
    """

    delta = stego.astype(np.int16) - cover.astype(np.int16)

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

    input_img = cover[:, :, 0] if cover.ndim == 3 else cover
    delta_2d = delta[:, :, 0] if delta.ndim == 3 else delta
    delta_arr = np.asarray(delta_2d, dtype=np.float64)

    results = []
    for name, cost_func in cost_functions.items():

        rho_raw = np.asarray(
            cost_func(input_img), dtype=np.float64
        )  # (2, H, W)

        # (2, H, W) → (H, W) kollabieren
        if name == "lsbr":
            even_mask = input_img % 2 == 0
            rho = np.where(even_mask, rho_raw[0], rho_raw[1])
        else:
            rho = rho_raw[0]  # symmetrisch, beide Kanäle gleich

        ternary = name not in ("lsbr",)
        est_lambda = estimate_parameters(delta_arr, rho, ternary=ternary)

        exponent = np.exp(-est_lambda * rho)
        if ternary:
            p_i = exponent / (1 + 2 * exponent)
            p0 = 1 - 2 * p_i
            est_M = float(np.sum(-(2 * p_i * np.log2(p_i + 1e-15) + p0 * np.log2(p0 + 1e-15))))
        else:
            est_M = 2.0 * float((delta_arr != 0).sum())

        if name == "lsbr":
            cover_flat = input_img.flatten()
            delta_flat = delta_arr.flatten()
            even_mask = cover_flat % 2 == 0
            impossible = np.any(
                (even_mask & (delta_flat == -1))
                | (~even_mask & (delta_flat == +1))
            )
            if impossible:
                log_lik = -np.inf
            else:
                log_lik = _log_likelihood(delta_arr ,exponent, ternary=False)
        else:
            log_lik  = _log_likelihood(delta_arr, exponent, ternary=True)

        results.append(
            {
                "method_name": name,
                "M": est_M,
                "lambda": est_lambda,
                "log_lik": log_lik,
            }
        )

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
    delta = (stego_dct.astype(np.int32) - cover_dct.astype(np.int32)).astype(np.float64)

    # Embed-Maske: nur non-zero AC, kein DC
    embed_mask_f5 = cover_dct != 0
    embed_mask_f5[:, :, 0, 0] = False

    zero_changed = bool(np.any((cover_dct == 0) & (delta != 0)))
    wrong_direction = bool(np.any(
        ((cover_dct > 0) & embed_mask_f5 & (delta == +1))
        | ((cover_dct < 0) & embed_mask_f5 & (delta == -1))
    ))

    all_methods = {
        "lsb":      (False, None),
        "nsf5":     (False, None),
        "f5":       (False, None),
        "juniward": (True, lambda: cl.juniward.compute_cost_adjusted(x0=cover_spatial, y0=cover_dct, qt=qtable)),
        "uerd":     (True, lambda: cl.uerd.compute_cost_adjusted(y0=cover_dct, qt=qtable)),
        "ebs":      (True, lambda: cl.ebs.compute_cost_adjusted(y0=cover_dct, qt=qtable)),
    }

    results = []
    for name, (ternary, cost_fn) in all_methods.items():

        if _is_impossible(name, cover_dct, delta, wrong_direction, zero_changed):
            results.append({"method_name": name, "M": 0.0, "lambda": 0.0, "log_lik": -np.inf})
            continue

        if cost_fn is None:  # uniform cost (lsb, nsf5, f5)
            rho = np.ones_like(cover_dct, dtype=np.float64)
            rho[cover_dct == 0] = 1e13
            rho[:, :, 0, 0] = 1e13
        else:
            rho_raw = np.asarray(cost_fn(), dtype=np.float64)
            rho = rho_raw[0] if rho_raw.ndim == 5 else rho_raw

        delta_fit = delta[embed_mask_f5]
        rho_fit = rho[embed_mask_f5]

        est_lambda = estimate_parameters(delta_fit, rho_fit, ternary=ternary)
        exponent = np.exp(-est_lambda * rho_fit)

        if ternary:
            p_i = exponent / (1 + 2 * exponent)
            p0 = 1 - 2 * p_i
            est_M = float(np.sum(-(2 * p_i * np.log2(p_i + 1e-15) + p0 * np.log2(p0 + 1e-15))))
        else:
            est_M = 2.0 * float((delta_fit != 0).sum())

        log_lik = _log_likelihood(delta_fit, exponent, ternary=ternary)
        results.append({"method_name": name, "M": est_M, "lambda": est_lambda, "log_lik": log_lik})

    best_res = max(results, key=lambda x: x["log_lik"])
    return {
        "method": best_res["method_name"],
        "M": best_res["M"],
        "lambda": best_res["lambda"],
        "all": results,
    }