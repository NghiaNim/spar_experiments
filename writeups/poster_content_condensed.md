# Poster content

Four figures in reading order:

1. `fig0_pipeline.png` (Methods)
2. `fig1_heatmap_panel.png` (Results, small)
3. `fig2_lookahead.png` (Results, big top)
4. `fig9_token_overlay.png` (Results, big bottom)

`fig9` is the new token-level overlay; produce it via `make rollout-overlay-decep && python3 tools/make_token_overlay.py`.

---

Title. **Misalignment Lens: Anticipating Misalignment from a Single Hidden State**

Authors. Nghia Nim, sraval@g.harvard.edu, Harvard / SPAR

---

Abstract. Future Lens (Pal et al., 2023) decodes upcoming tokens from a single hidden state; we ask whether the same trick works for upcoming *misaligned behavior*. We train linear probes on residual-stream activations of Llama-3.2-1B-abliterated to predict, from the activation at token t, whether token t+k belongs to a misaligned span. Across sandbagging, instrumental deception, and manipulation, commitment is linearly decodable from a shared **mid-stack band** several tokens before the token is emitted, in *behavior-specific directions* that share the band but not their orientation.

Introduction. We want a function from the activation at token t to the probability of misalignment within the next k tokens, with positive lead time and behavior breadth. Linear probes are the natural starting point. The question is whether they read misalignment-specific lookahead or only the presence of a misaligned persona.

Methods. [fig0] Matched hack and honest personas drive self-generation; GPT-5.4-mini labels commitment-span tokens; a linear probe is trained per layer L and offset k.

Mid-stack band. [fig1] The strongest probe sits in **layers 7 to 13** for all three behaviors, ahead of the misaligned token. The band reproduces across seeds; the single best cell is seed-stable only for deception. *Cite the band, not the cell.*

Lookahead horizon. [fig2] The probe degrades smoothly with the offset k. Deception keeps the most signal at large k; reward-hacking pilots stay highest of all but at far lower base rate.

Token-level reading. [fig9] On fresh rollouts, the probe lights up several tokens before the labeled-lie token, with substantial lead time. The same probe also lights up on honest rollouts, which is the deployment caveat in one image: catch ≈ false-positive rate at the F1-tuned threshold.

Probe geometry. The three probes share the layer band but point in *unrelated directions* at the random-vector noise floor: **co-located linear decoders, not a shared misalignment direction**. A small MLP at the same cells matches the linear probe, so the representation is *linearly accessible*.

Discussion. Probes support **offline auditing** with known persona; a working **deployment detector** is still open.

Future work.
1. **Causal interventions**: patch the probe direction at the chosen layer and test whether suppressing it prevents the misaligned commit; amplifying it should induce one in honest contexts.
2. **Scale**: rerun on larger models (Llama-3-8B, Qwen-7B, Gemma) to test whether the mid-stack band is an *architectural property*.
3. **Sparse autoencoder features**: probe on SAE-decomposed features instead of raw residuals, to break the *unrelated directions* result into named latent units.
4. **Cross-behavior probe**: train on the union of the three behaviors and test whether it discovers a *commitment to misaligned action* feature, or collapses onto one of the per-behavior directions.

References. Pal, K. et al., Future Lens: Anticipating Subsequent Tokens from a Single Hidden State (CoNLL 2023). Paper draft: writeups/paper_final.md. Model: huihui-ai/Llama-3.2-1B-Instruct-abliterated. Labeler: OpenAI gpt-5.4-mini. Compute: Modal.
