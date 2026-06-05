"""
Blind steganalysis attack for spatial and JPEG domain images.

Estimates the embedding method and embedded message length of a stego image
by fitting a statistical embedding model to the observed change pattern.
For each candidate method, the Lagrange multiplier lambda is optimized
via multiplicative update to match the theoretical change rate to the
observed change rate. The method with the highest log-likelihood is returned
together with an estimate of the embedded message length in bits.

The Lambda optimization is grounded in the monotonicity of the Gibbs
distribution entropy w.r.t. lambda [1].

**Generic** (see :func:`attack`)

Auto-detects spatial vs. JPEG domain based on input shape.

**Spatial domain** (see :func:`attack_spatial`)

Supported methods: ``hill``, ``hugo``, ``suniward``, ``wow``, ``lsbm``, ``lsbr``.

**JPEG domain** (see :func:`attack_jpeg`)

Supported methods: ``juniward``, ``uerd``, ``ebs``, ``nsf5``, ``f5``, ``lsb``.

.. note::

    The JPEG attack is experimental. Detection accuracy varies significantly
    with embedding rate (alpha) and image content.
    ``nsf5`` and ``f5`` are statistically indistinguishable with this approach —
    both algorithms use uniform-cost unidirectional embedding and produce
    nearly identical change patterns. ``f5`` images will typically be
    classified as ``nsf5``.

.. rubric:: References

.. [1] T. Filler, J. Judas, and J. Fridrich, "Minimizing Additive Distortion
   in Steganography Using Syndrome-Trellis Codes," *IEEE Trans. Inf. Forensics
   Security*, vol. 6, no. 3, pp. 920--935, 2011.

:author: Jonas Feierabend
:affiliation: University of Innsbruck
"""

from ._attack import attack, attack_spatial, attack_jpeg  # noqa: F401