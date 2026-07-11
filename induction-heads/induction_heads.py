# %% [markdown]
# # Induction heads in GPT-2 small
#
# Work through the phases in PLAN.md. Each `# %%` block is a runnable cell in VS Code
# or Jupyter. Phases 1-4 give a complete result; phase 5 is an optional stretch. Don't
# just copy from tutorials — form a hypothesis, then test it. Save the key plots to
# `figures/` for the write-up.

# %% Setup — load GPT-2 small and confirm the environment works
import torch
from transformer_lens import HookedTransformer

device = "cuda" if torch.cuda.is_available() else "cpu"
model = HookedTransformer.from_pretrained("gpt2", device=device)
model.eval()
print(f"GPT-2 small: {model.cfg.n_layers} layers x {model.cfg.n_heads} heads, device={device}")

# %% Phase 1 — orient: run with cache, peek at an attention pattern
text = "The cat sat on the mat. The cat sat on the"
tokens = model.to_tokens(text)
logits, cache = model.run_with_cache(tokens)
print("predicted next token:", repr(model.to_string(logits[0, -1].argmax())))

# TODO: visualize an attention pattern to get a feel for the tools.
#   import circuitsvis as cv
#   cv.attention.attention_patterns(
#       tokens=model.to_str_tokens(text),
#       attention=cache["pattern", 0][0],   # layer 0: [head, query, key]
#   )
# `cache["pattern", layer]` is the post-softmax attention. Explore a few layers.

# %% Phase 2 — show the phenomenon: in-context copying on repeated random tokens
# TODO:
#   - Build a sequence: [BOS] + random_tokens + (the SAME random_tokens again).
#     e.g. rand = torch.randint(0, model.cfg.d_vocab, (1, seq_len))
#          seq  = torch.cat([bos, rand, rand], dim=1)
#   - Run the model and compute per-position cross-entropy loss on predicting the next token.
#   - Plot loss vs position. You should see loss DROP sharply on the second copy:
#     the model is copying from earlier in the context. Save the plot to figures/.

# %% Phase 3 — locate the mechanism: an "induction score" per attention head
# TODO:
#   - On the repeated sequence, an INDUCTION head at query position i attends to key
#     position (i - seq_len + 1): the token that came *right after* the previous
#     occurrence of the current token.
#   - For each (layer, head), average the attention mass sitting on that offset
#     (the "induction stripe") -> an induction score. Use cache["pattern", layer][0, head].
#   - Rank heads by score, identify the top few, and visualize their patterns to confirm.

# %% Phase 4 — the core interp move: prove causality by ablation
# TODO:
#   - Ablate the top induction head(s): hook the head's output z and zero (or mean-)
#     ablate it, then re-measure in-context copying loss.
#       from transformer_lens import utils
#       def ablate_head(z, hook, head):        # z: [batch, pos, head, d_head]
#           z[:, :, head, :] = 0.0
#           return z
#       model.run_with_hooks(seq, fwd_hooks=[(utils.get_act_name("z", layer),
#                                             partial(ablate_head, head=head))])
#   - Show copying performance collapses when you ablate induction heads, but NOT when
#     you ablate random heads. That contrast is the result: these heads are *causally*
#     responsible, not merely correlated.

# %% Phase 5 (stretch) — make it a circuit: previous-token heads feed the induction heads
# TODO:
#   - Find "previous-token heads" (attend from i to i-1) in earlier layers. These write
#     the "what came before me" info that induction heads read (K-composition).
#   - Ablate/patch them and show induction degrades -> a two-step circuit, not one head.
#   - See Anthropic's "A Mathematical Framework for Transformer Circuits" for the theory.

# %% Phase 6 — write up
# TODO: drop your key plots in figures/ and write the story in the README as
# hypothesis -> test -> result, with honest limitations.
