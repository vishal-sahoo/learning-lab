# Induction heads in GPT-2 small

Reverse-engineering how GPT-2 small copies from its context: finding the attention heads
that implement in-context copying, and proving, causally, that they are responsible.

When GPT-2 sees `... A B ... A`, it predicts `B`: it copies from earlier in the context.
This in-context copying is the seed of in-context learning, and a small transformer
implements it with a specific, findable mechanism called **induction heads**. The goal is
to answer, with evidence: *which components of GPT-2 small implement in-context copying,
and how do we know they cause it?*

## Background

- **Induction head** — an attention head that, at the current token, attends back to the
  token right after the *previous* occurrence of the current token, and copies it forward.
  On a repeated sequence, it attends from position `i` to `i - seq_len + 1`.
- **Previous-token head** — an earlier-layer head attending from each token to the one
  before it. It writes "what came before me" into the residual stream, which the induction
  head then reads (this composition is *K-composition*). Together they form the induction
  circuit.
- **Induction score** — how much of a head's attention sits on the induction offset; a
  cheap way to rank all heads and find the candidates.
- **Ablation** — knock out a component and measure the damage. If copying collapses when
  an induction head is ablated but not when a random head is, that head is *causally*
  responsible.

## Setup

```bash
cd induction-heads
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

Open [`induction_heads.py`](induction_heads.py) in VS Code or Jupyter (each `# %%` block is
a cell), or paste the cells into Google Colab with a GPU. GPT-2 small (124M) also runs fine
on CPU.

## Results

_(filled in as the experiments run)_

- **Hypothesis:**
- **Method:**
- **Findings:**
- **Limitations / open questions:**

---

See **[PLAN.md](PLAN.md)** for the step-by-step plan and study resources.
