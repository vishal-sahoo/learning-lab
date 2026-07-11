# Plan — induction heads in GPT-2 small

The step-by-step plan, and the resources to study alongside it. Work through the phases in
[`induction_heads.py`](induction_heads.py). Phases 1-4 give a complete result; phase 5 is
an optional deeper extension.

## Phases

1. **Orient** — load GPT-2 small in TransformerLens, run it with `run_with_cache`, and
   visualize an attention pattern. Get comfortable with hooks and the residual stream.
2. **See the phenomenon** — build a repeated-random-tokens sequence, plot per-position
   loss, and watch it drop on the second copy (the model copying from context).
3. **Locate the mechanism** — compute an induction score for every head, find the top
   induction heads, and visualize them to confirm the behavior.
4. **Prove causality** — ablate the induction heads and show copying collapses, while
   ablating random heads doesn't. This contrast is the core result.
5. **Stretch: the circuit** — find the previous-token heads that feed the induction heads,
   and show that breaking the composition also breaks induction.
6. **Write up** — save the key plots to `figures/`, and fill in the Results section of the
   README as hypothesis → test → conclusion, with honest limitations.

## What "done" looks like

- Plots in `figures/`: the per-position loss drop (phase 2), an induction-score heatmap
  over all heads (phase 3), and the ablation comparison (phase 4).
- The README Results section written up. A clean phases 1-4 result is already complete on
  its own.

## Resources

**Start here (guided path):**
- **ARENA — Chapter 1, section [1.2] "Intro to Mechanistic Interpretability"** — walks
  TransformerLens and building induction heads step by step; the best single resource for
  this. <https://arena3-chapter1-transformer-interp.streamlit.app/> (or find it from
  <https://www.arena.education/>).
- **Neel Nanda — Getting Started in Mech Interp** — orientation and concrete first steps,
  plus TransformerLens tutorial notebooks and YouTube walkthroughs.
  <https://www.neelnanda.io/mechanistic-interpretability/getting-started>

**Theory:**
- Anthropic — **"In-context Learning and Induction Heads"** (the paper this is based on):
  <https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html>
- Anthropic — **"A Mathematical Framework for Transformer Circuits"** (residual stream,
  Q/K/V, composition — the theory behind the circuit, needed for phase 5):
  <https://transformer-circuits.pub/2021/framework/index.html>

**Tools:**
- **TransformerLens**: <https://github.com/TransformerLensOrg/TransformerLens> · docs
  <https://transformerlensorg.github.io/TransformerLens/> (see the "Main Demo" notebook).
- **CircuitsVis** (attention visualizations):
  <https://github.com/TransformerLensOrg/CircuitsVis>
