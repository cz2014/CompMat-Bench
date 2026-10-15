# Interface distribution of a slab Hall response

The supplied `surface_model.py` defines a SnTe (110) slab with interface
exchange. Use its default parameters and 40 atomic planes. The occupied
subspace consists of the lowest 241 eigenstates at each momentum.

Using the occupied-space real-space Chern-density operator, determine the
layer-resolved contribution to the Hall response over the
momentum window $q\in[-0.04125,0.04125]$ and
$k_z\in[0.8175,1.1825]$, using the physical coordinates defined in the module.
Include the finite-gap contributions throughout this window. Report values
in units of $e^2/h$.

Let $C_l$ denote the signed layer contribution, with layer membership and
interface-first ordering supplied by `plane_orbitals`. Report the magnitude
of the cumulative contribution from the first ten planes and the magnitude
profile from the interface to the slab midpoint. The cumulative magnitude
is $|\sum_{l=1}^{10} C_l|$; the profile contains $|C_l|$ for layers 1 through 20.

Write `answer.json` using this schema template. Replace the typed placeholders
with your computed finite real values:

```text
{
  "abs_chern_within_10_layers": <float>,
  "C_layer_interface_abs": [<float>, ...]
}
```

The profile contains 20 finite real values in the layer order stated above. The requested quantities are finite-window response coefficients.
