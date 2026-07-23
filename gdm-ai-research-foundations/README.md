# Google DeepMind — AI Research Foundations

My completed lab notebooks from the [AI Research Foundations](https://www.skills.google/paths/3135)
path on Google Skills, built with Google DeepMind. The course works bottom-up: start from
plain probability over tokens, build the machinery to turn text into vectors, then build
attention and the transformer that replaces the n-gram guesswork.

Each notebook is the Colab as I ran it, with my code in the exercise cells and the outputs
kept, so the results are readable without re-running anything. Notebooks are unmodified
otherwise and remain subject to the licence in the
[google-deepmind/ai-foundations](https://github.com/google-deepmind/ai-foundations) README.

## Labs

### 1 — Small language models ([`lab1-slm/`](lab1-slm/))

Building up to a working SLM, starting from the idea that a language model is just a
probability distribution over the next token.

| | Notebook |
|---|---|
| 1.1 | [Create your own probability distribution](lab1-slm/Copy%20of%20gdm_lab_1_1_create_your_own_probability_distribution.ipynb) |
| 1.2 | [Experiment with n-gram models](lab1-slm/Copy%20of%20gdm_lab_1_2_experiment_with_n_gram_models.ipynb) |
| 1.3 | [Compare n-gram models and transformer language models](lab1-slm/Copy%20of%20gdm_lab_1_3_compare_n_gram_models_and_transformer_language_models.ipynb) |
| 1.4 | [Prepare the dataset for training a SLM](lab1-slm/Copy%20of%20gdm_lab_1_4_prepare_the_dataset_for_training_a_slm.ipynb) |
| 1.5 | [Train your own small language model](lab1-slm/Copy%20of%20gdm_lab_1_5_train_your_own_small_language_model.ipynb) |

### 2 — Representing data ([`lab2-represent-data/`](lab2-represent-data/))

How raw text becomes model input: cleaning, tokenization from characters up to BPE
subwords, and what the resulting embeddings encode — then retraining the SLM on my own
tokenizer.

| | Notebook |
|---|---|
| 2.1 | [Preprocess data](lab2-represent-data/Copy%20of%20gdm_lab_2_1_preprocess_data.ipynb) |
| 2.2 | [Tokenize texts into characters and words](lab2-represent-data/Copy%20of%20gdm_lab_2_2_tokenize_texts_into_characters_and_words.ipynb) |
| 2.3 | [Tokenize texts into subword tokens](lab2-represent-data/Copy%20of%20gdm_lab_2_3_tokenize_texts_into_subword_tokens.ipynb) |
| 2.4 | [Implement a BPE tokenizer](lab2-represent-data/Copy%20of%20gdm_lab_2_4_implement_a_bpe_tokenizer.ipynb) |
| 2.5 | [Experiment with embeddings](lab2-represent-data/Copy%20of%20gdm_lab_2_5_experiment_with_embeddings.ipynb) |
| 2.6 | [Train an SLM with your BPE tokenizer](lab2-represent-data/Copy%20of%20gdm_lab_2_6_train_an_slm_with_your_bpe_tokenizer.ipynb) |

### 4 — The transformer ([`lab4-transformer/`](lab4-transformer/))

Attention from the inside: visualizing what heads attend to, implementing the attention
equation by hand, and the two things attention alone can't do — position and parameter
budget.

| | Notebook |
|---|---|
| 4.1 | [Attention visualization](lab4-transformer/Copy%20of%20gdm_lab_4_1_attention_visualization.ipynb) |
| 4.2 | [Implement attention equation](lab4-transformer/Copy%20of%20gdm_lab_4_2_implement_attention_equation.ipynb) |
| 4.3 | [Implement attention equation 2](lab4-transformer/Copy%20of%20gdm_lab_4_3_implement_attention_equation_2.ipynb) |
| 4.4 | [Positional embeddings](lab4-transformer/Copy%20of%20gdm_lab_4_4_positional_embeddings.ipynb) |
| 4.5 | [Reflection on trainable parameters](lab4-transformer/Copy%20of%20gdm_lab_4_5_reflection_on_trainable_parameters.ipynb) |

Units 1, 2 and 4 are what's here; unit 3 had no notebooks to import.

## Running these

They're Colab notebooks — the least friction is to open one in
[Colab](https://colab.research.google.com/) (`File → Upload notebook`) and run it there,
which is also where the GPU for the training labs comes from. GitHub renders them
read-only, which is enough if you only want to see the outputs.

---

Where this goes next: unit 4 is the direct lead-in to
[induction-heads](../induction-heads/), which takes the same attention machinery and asks
what a *trained* model actually does with it.
