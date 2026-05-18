# Shortcuts for the probe pipeline. Run `make help` for the menu.
#
# Five experiments:
#   - reward-hack probe (3 task variants: oneshot / explicit / syco)
#   - sandbagging probe (single task: sandbag, with v1/v2 label variants)
#   - manipulation probe (single task: manipulation, with v1/v2 label variants)
#   - deception probe (single task: deception, with v1/v2 label variants)
#   - profanity probe (legacy, prefixed `prof-`)
#
# The most common workflow once data is already on the Modal volume:
#   make resume-oneshot     # re-label and re-probe
#   make resume-sandbag
#
# Override any tunable on the command line, e.g.
#   make label-oneshot WORKERS=2 LABELER_MODELS=gpt-5.4
#   make probe-syco EPOCHS=400 NEG_PER_POS=5
#   make all-sandbag SAMPLES_PER_PROMPT=6 MAX_OFFSET=15

# ----- tunables -----------------------------------------------------------
LABELER_MODELS     ?= gpt-5.4-mini,gpt-5.4
WORKERS            ?= 4
PER_MODEL_ATTEMPTS ?= 2
MAX_TOTAL_WAIT     ?= 180
SAMPLES_PER_PROMPT ?= 4
MAX_OFFSET         ?= 10
EPOCHS             ?= 200
NEG_PER_POS        ?= 10
RUN_NAME           ?=

# ----- internal -----------------------------------------------------------
RH := modal run reward_hack_experiment/run.py
SB := modal run sandbag_experiment/run.py
MN := modal run manipulation_experiment/run.py
DE := modal run deception_experiment/run.py
P  := modal run probe_experiment/run.py

LABEL_FLAGS := --labeler-models "$(LABELER_MODELS)" --label-workers $(WORKERS) \
               --per-model-attempts $(PER_MODEL_ATTEMPTS) \
               --max-total-wait $(MAX_TOTAL_WAIT)

PROBE_FLAGS := --max-offset $(MAX_OFFSET) --num-epochs $(EPOCHS) \
               --neg-per-pos $(NEG_PER_POS)
ifneq ($(strip $(RUN_NAME)),)
PROBE_FLAGS += --run-name $(RUN_NAME)
endif

MODEL_FLAGS := --samples-per-prompt $(SAMPLES_PER_PROMPT)

.DEFAULT_GOAL := help

# ----- substring_oneshot --------------------------------------------------
.PHONY: data-oneshot label-oneshot probe-oneshot resume-oneshot all-oneshot pull-oneshot

data-oneshot:
	$(RH) --task substring_oneshot --stage model $(MODEL_FLAGS)

label-oneshot:
	$(RH) --task substring_oneshot --stage label $(LABEL_FLAGS)

probe-oneshot:
	$(RH) --task substring_oneshot --stage probe $(PROBE_FLAGS)

resume-oneshot: label-oneshot probe-oneshot

all-oneshot:
	$(RH) --task substring_oneshot --stage all $(MODEL_FLAGS) $(LABEL_FLAGS) $(PROBE_FLAGS)

pull-oneshot:
	$(RH) --task substring_oneshot --stage download

# ----- substring_explicit -------------------------------------------------
.PHONY: data-explicit label-explicit probe-explicit resume-explicit all-explicit pull-explicit

data-explicit:
	$(RH) --task substring_explicit --stage model $(MODEL_FLAGS)

label-explicit:
	$(RH) --task substring_explicit --stage label $(LABEL_FLAGS)

probe-explicit:
	$(RH) --task substring_explicit --stage probe $(PROBE_FLAGS)

resume-explicit: label-explicit probe-explicit

all-explicit:
	$(RH) --task substring_explicit --stage all $(MODEL_FLAGS) $(LABEL_FLAGS) $(PROBE_FLAGS)

pull-explicit:
	$(RH) --task substring_explicit --stage download

# ----- sycophancy ---------------------------------------------------------
.PHONY: data-syco label-syco probe-syco resume-syco all-syco pull-syco

data-syco:
	$(RH) --task sycophancy --stage model $(MODEL_FLAGS)

label-syco:
	$(RH) --task sycophancy --stage label $(LABEL_FLAGS)

probe-syco:
	$(RH) --task sycophancy --stage probe $(PROBE_FLAGS)

resume-syco: label-syco probe-syco

all-syco:
	$(RH) --task sycophancy --stage all $(MODEL_FLAGS) $(LABEL_FLAGS) $(PROBE_FLAGS)

pull-syco:
	$(RH) --task sycophancy --stage download

# ----- combos across all 3 reward-hack tasks ------------------------------
.PHONY: data-all-rh resume-all-rh all-all-rh pull-all-rh

data-all-rh:    data-oneshot data-explicit data-syco
resume-all-rh:  resume-oneshot resume-explicit resume-syco
all-all-rh:     all-oneshot all-explicit all-syco
pull-all-rh:    pull-oneshot pull-explicit pull-syco

# ----- sandbagging --------------------------------------------------------
.PHONY: data-sandbag label-sandbag probe-sandbag resume-sandbag all-sandbag pull-sandbag

data-sandbag:
	$(SB) --task sandbag --stage model $(MODEL_FLAGS)

label-sandbag:
	$(SB) --task sandbag --stage label $(LABEL_FLAGS)

probe-sandbag:
	$(SB) --task sandbag --stage probe $(PROBE_FLAGS)

resume-sandbag: label-sandbag probe-sandbag

all-sandbag:
	$(SB) --task sandbag --stage all $(MODEL_FLAGS) $(LABEL_FLAGS) $(PROBE_FLAGS)

pull-sandbag:
	$(SB) --task sandbag --stage download

# ----- sandbag v2 (transition-only labels) --------------------------------
# Same corpus + activations as v1 — switching variants only re-runs label
# + probe. Outputs land at labels_transition.json / samples_transition.jsonl
# / results_transition/ on the volume so v1 results stay intact for
# side-by-side comparison.
.PHONY: label-sandbag-v2 probe-sandbag-v2 samples-sandbag-v2 \
        resume-sandbag-v2 all-sandbag-v2 pull-sandbag-v2

label-sandbag-v2:
	$(SB) --task sandbag --stage label --label-variant transition $(LABEL_FLAGS)

probe-sandbag-v2:
	$(SB) --task sandbag --stage probe --label-variant transition $(PROBE_FLAGS)

samples-sandbag-v2:
	$(SB) --task sandbag --stage samples --label-variant transition

resume-sandbag-v2: label-sandbag-v2 probe-sandbag-v2

# all-sandbag-v2 does NOT regen the model stage (which is variant-agnostic).
# Run `make data-sandbag` first if you don't have a corpus yet.
all-sandbag-v2: label-sandbag-v2 probe-sandbag-v2

pull-sandbag-v2:
	$(SB) --task sandbag --stage download --label-variant transition

# ----- manipulation -------------------------------------------------------
.PHONY: data-manip label-manip probe-manip resume-manip all-manip pull-manip

data-manip:
	$(MN) --task manipulation --stage model $(MODEL_FLAGS)

label-manip:
	$(MN) --task manipulation --stage label $(LABEL_FLAGS)

probe-manip:
	$(MN) --task manipulation --stage probe $(PROBE_FLAGS)

resume-manip: label-manip probe-manip

all-manip:
	$(MN) --task manipulation --stage all $(MODEL_FLAGS) $(LABEL_FLAGS) $(PROBE_FLAGS)

pull-manip:
	$(MN) --task manipulation --stage download

# ----- manipulation v2 (transition-only labels) ---------------------------
# Same corpus + activations as v1 — switching variants only re-runs label
# + probe. Outputs land at labels_transition.json / samples_transition.jsonl
# / results_transition/ on the volume so v1 results stay intact for
# side-by-side comparison.
.PHONY: label-manip-v2 probe-manip-v2 samples-manip-v2 \
        resume-manip-v2 all-manip-v2 pull-manip-v2

label-manip-v2:
	$(MN) --task manipulation --stage label --label-variant transition $(LABEL_FLAGS)

probe-manip-v2:
	$(MN) --task manipulation --stage probe --label-variant transition $(PROBE_FLAGS)

samples-manip-v2:
	$(MN) --task manipulation --stage samples --label-variant transition

resume-manip-v2: label-manip-v2 probe-manip-v2

# all-manip-v2 does NOT regen the model stage (which is variant-agnostic).
# Run `make data-manip` first if you don't have a corpus yet.
all-manip-v2: label-manip-v2 probe-manip-v2

pull-manip-v2:
	$(MN) --task manipulation --stage download --label-variant transition

# ----- deception ----------------------------------------------------------
.PHONY: data-decep label-decep probe-decep resume-decep all-decep pull-decep

data-decep:
	$(DE) --task deception --stage model $(MODEL_FLAGS)

label-decep:
	$(DE) --task deception --stage label $(LABEL_FLAGS)

probe-decep:
	$(DE) --task deception --stage probe $(PROBE_FLAGS)

resume-decep: label-decep probe-decep

all-decep:
	$(DE) --task deception --stage all $(MODEL_FLAGS) $(LABEL_FLAGS) $(PROBE_FLAGS)

pull-decep:
	$(DE) --task deception --stage download

# ----- deception v2 (transition-only labels) ------------------------------
# Same corpus + activations as v1 — switching variants only re-runs label
# + probe. Outputs land at labels_transition.json / samples_transition.jsonl
# / results_transition/ on the volume so v1 results stay intact for
# side-by-side comparison.
.PHONY: label-decep-v2 probe-decep-v2 samples-decep-v2 \
        resume-decep-v2 all-decep-v2 pull-decep-v2

label-decep-v2:
	$(DE) --task deception --stage label --label-variant transition $(LABEL_FLAGS)

probe-decep-v2:
	$(DE) --task deception --stage probe --label-variant transition $(PROBE_FLAGS)

samples-decep-v2:
	$(DE) --task deception --stage samples --label-variant transition

resume-decep-v2: label-decep-v2 probe-decep-v2

# all-decep-v2 does NOT regen the model stage (which is variant-agnostic).
# Run `make data-decep` first if you don't have a corpus yet.
all-decep-v2: label-decep-v2 probe-decep-v2

pull-decep-v2:
	$(DE) --task deception --stage download --label-variant transition

# ===== Follow-up probes (post-paper v0): completion-level, per-stratum, ===
# ===== MLP sanity, probe-direction similarity.                          ===
#
# All four operate against existing on-volume activations + labels.
# Completion-level + per-stratum + MLP each have a `--label-variant
# {full,transition}` flag (defaults to transition since v2 is the
# headline). Probe similarity is a local script that reads downloaded
# probe_weights.pt files.
#
# Override at the command line, e.g.
#   make completion-decep-v2 PREFIX_LENGTHS=1,5,10
#   make mlp-sanity-decep-v2 MLP_LAYER=7 MLP_OFFSETS=0,3,5
#   make per-stratum-decep-v2 STRATUM_FIELD=claim_category

PREFIX_LENGTHS    ?= 1,3,5,10,20
MLP_LAYER         ?= 9
MLP_LAYER_MANIP   ?= 12
MLP_OFFSETS       ?= 0,5
MLP_HIDDEN        ?= 256
SEED              ?= 0
STRATUM_FIELD_DECEP ?= claim_category
STRATUM_FIELD_MANIP ?= tactic
STRATUM_FIELD_SB    ?= sandbag_mode

# ----- completion-level (aggregate) probe -----------------------------
# Mean-pool activations over the first N tokens of each completion, predict
# per-completion outcome. Closes the manipulation v2 negative result.
.PHONY: completion-sandbag-v2 completion-manip-v2 completion-decep-v2

completion-sandbag-v2:
	$(SB) --task sandbag --stage completion-probe --label-variant transition \
	      --prefix-lengths $(PREFIX_LENGTHS) --seed $(SEED) \
	      --neg-per-pos $(NEG_PER_POS) --num-epochs $(EPOCHS)

completion-manip-v2:
	$(MN) --task manipulation --stage completion-probe --label-variant transition \
	      --prefix-lengths $(PREFIX_LENGTHS) --seed $(SEED) \
	      --neg-per-pos $(NEG_PER_POS) --num-epochs $(EPOCHS)

completion-decep-v2:
	$(DE) --task deception --stage completion-probe --label-variant transition \
	      --prefix-lengths $(PREFIX_LENGTHS) --seed $(SEED) \
	      --neg-per-pos $(NEG_PER_POS) --num-epochs $(EPOCHS)

# Same but on v1 (full-span labels) — useful for manipulation since v1 is
# its headline.
.PHONY: completion-manip-v1 completion-sandbag-v1 completion-decep-v1

completion-manip-v1:
	$(MN) --task manipulation --stage completion-probe \
	      --prefix-lengths $(PREFIX_LENGTHS) --seed $(SEED) \
	      --neg-per-pos $(NEG_PER_POS) --num-epochs $(EPOCHS)

completion-sandbag-v1:
	$(SB) --task sandbag --stage completion-probe \
	      --prefix-lengths $(PREFIX_LENGTHS) --seed $(SEED) \
	      --neg-per-pos $(NEG_PER_POS) --num-epochs $(EPOCHS)

completion-decep-v1:
	$(DE) --task deception --stage completion-probe \
	      --prefix-lengths $(PREFIX_LENGTHS) --seed $(SEED) \
	      --neg-per-pos $(NEG_PER_POS) --num-epochs $(EPOCHS)

# ----- per-stratum (per-category / per-tactic / per-mode) -------------
# Filter hack-class completions by the stratum field (claim_category for
# deception, tactic for manipulation, sandbag_mode for sandbag) and run
# the token-level sweep on each stratum separately.
.PHONY: per-stratum-sandbag-v2 per-stratum-manip-v2 per-stratum-decep-v2

per-stratum-sandbag-v2:
	$(SB) --task sandbag --stage per-stratum-probe --label-variant transition \
	      --stratum-field $(STRATUM_FIELD_SB) --seed $(SEED) \
	      --max-offset $(MAX_OFFSET) --neg-per-pos $(NEG_PER_POS) --num-epochs $(EPOCHS)

per-stratum-manip-v2:
	$(MN) --task manipulation --stage per-stratum-probe --label-variant transition \
	      --stratum-field $(STRATUM_FIELD_MANIP) --seed $(SEED) \
	      --max-offset $(MAX_OFFSET) --neg-per-pos $(NEG_PER_POS) --num-epochs $(EPOCHS)

per-stratum-decep-v2:
	$(DE) --task deception --stage per-stratum-probe --label-variant transition \
	      --stratum-field $(STRATUM_FIELD_DECEP) --seed $(SEED) \
	      --max-offset $(MAX_OFFSET) --neg-per-pos $(NEG_PER_POS) --num-epochs $(EPOCHS)

# ----- MLP-vs-linear sanity check -------------------------------------
# Trains a 2-layer MLP and a linear probe on the same (layer, offset)
# cell. Tells us whether linear is the right tool or if non-linearity
# buys lift.
.PHONY: mlp-sanity-sandbag-v2 mlp-sanity-manip-v2 mlp-sanity-decep-v2

mlp-sanity-sandbag-v2:
	$(SB) --task sandbag --stage mlp-sanity --label-variant transition \
	      --mlp-layer $(MLP_LAYER) --mlp-offsets $(MLP_OFFSETS) \
	      --mlp-hidden $(MLP_HIDDEN) --num-epochs $(EPOCHS) \
	      --neg-per-pos $(NEG_PER_POS) --seed $(SEED)

mlp-sanity-manip-v2:
	$(MN) --task manipulation --stage mlp-sanity --label-variant transition \
	      --mlp-layer $(MLP_LAYER_MANIP) --mlp-offsets $(MLP_OFFSETS) \
	      --mlp-hidden $(MLP_HIDDEN) --num-epochs $(EPOCHS) \
	      --neg-per-pos $(NEG_PER_POS) --seed $(SEED)

mlp-sanity-decep-v2:
	$(DE) --task deception --stage mlp-sanity --label-variant transition \
	      --mlp-layer $(MLP_LAYER) --mlp-offsets $(MLP_OFFSETS) \
	      --mlp-hidden $(MLP_HIDDEN) --num-epochs $(EPOCHS) \
	      --neg-per-pos $(NEG_PER_POS) --seed $(SEED)

# ----- Probe-direction similarity (local) -----------------------------
# Reads probe_weights.pt files downloaded by pull-*-v2 and computes
# pairwise cosine similarity. Run after the v2 probe sweeps + their
# pull steps have completed.
.PHONY: probe-similarity probe-similarity-sweep

probe-similarity:
	python3 tools/probe_similarity.py --layer $(MLP_LAYER) --offset 0 \
	    --label-variant transition --out results/probe_similarity_L$(MLP_LAYER)_k0.json

probe-similarity-sweep:
	python3 tools/probe_similarity.py --sweep --label-variant transition \
	    --out results/probe_similarity_sweep.json

# ----- meta: run everything in the follow-up bundle -------------------
# This target also re-runs probe-{exp}-v2 first to regenerate probe_weights.pt
# (the new sweep_layers_and_offsets saves probe weights as a side effect; the
# existing v2 results on the volume predate this and don't have them, which
# probe-similarity needs locally).
#
# Modal time budget (serial): ~3 probe re-runs (~15min each) + 3 completion
# (~5min each) + 3 per-stratum (~10min each) + 3 mlp-sanity (~3min each) =
# ~100 minutes on L4. Pulls + local similarity adds ~5 min.
#
# If you don't want to re-run the v2 token probes, use `followups-no-probes`
# instead and skip probe-similarity (or run it after probe-{exp}-v2 manually).
.PHONY: followups-all followups-no-probes
followups-all: probe-sandbag-v2 probe-manip-v2 probe-decep-v2 \
               completion-sandbag-v2 completion-manip-v2 completion-decep-v2 \
               completion-manip-v1 \
               per-stratum-sandbag-v2 per-stratum-manip-v2 per-stratum-decep-v2 \
               mlp-sanity-sandbag-v2 mlp-sanity-manip-v2 mlp-sanity-decep-v2 \
               pull-sandbag-v2 pull-manip-v2 pull-decep-v2 \
               probe-similarity

followups-no-probes: completion-sandbag-v2 completion-manip-v2 completion-decep-v2 \
               completion-manip-v1 \
               per-stratum-sandbag-v2 per-stratum-manip-v2 per-stratum-decep-v2 \
               mlp-sanity-sandbag-v2 mlp-sanity-manip-v2 mlp-sanity-decep-v2 \
               pull-sandbag-v2 pull-manip-v2 pull-decep-v2

# Compute-only bundle: skip every download + the local similarity step.
# Use this when network is unreliable; pair with `make pull-followups`
# afterwards (when network is stable) to bring results down + run similarity.
.PHONY: followups-compute pull-followups
followups-compute: probe-sandbag-v2 probe-manip-v2 probe-decep-v2 \
               completion-sandbag-v2 completion-manip-v2 completion-decep-v2 \
               completion-manip-v1 \
               per-stratum-sandbag-v2 per-stratum-manip-v2 per-stratum-decep-v2 \
               mlp-sanity-sandbag-v2 mlp-sanity-manip-v2 mlp-sanity-decep-v2

# Pull-only bundle: pull all three experiments' v2 artifacts and then run the
# local probe-similarity script. Run this after `make followups-compute` when
# network has recovered, or any time you want to refresh local copies.
# Each pull is non-fatal so a transient failure on one experiment doesn't
# stop the others.
pull-followups:
	-$(MAKE) pull-sandbag-v2
	-$(MAKE) pull-manip-v2
	-$(MAKE) pull-decep-v2
	-$(MAKE) probe-similarity

# ===== Multi-seed reruns of the v2 probe sweep ============================
# Re-runs the existing v2 probe with seeds 0..4, writing to per-seed
# subdirs (np10_mo10_ep200_seedN). Then aggregates locally to mean ± 95%
# bootstrap CI. Closes the highest-priority reviewer concern from the
# critique pile (single-seed claims).
#
# Modal time budget: 5 seeds × ~15min on L4 = ~75 min per experiment;
# all three serial is ~3.5 hours. Each seed runs both `all/` and
# `<exp>_prompts/` subsets so the cells are fully populated.
.PHONY: multi-seed-sandbag-v2 multi-seed-manip-v2 multi-seed-decep-v2 \
        multi-seed-all-v2 multi-seed-aggregate

multi-seed-sandbag-v2:
	$(SB) --task sandbag --stage multi-seed --label-variant transition \
	      --multi-seeds 0,1,2,3,4 --max-offset $(MAX_OFFSET) \
	      --num-epochs $(EPOCHS) --neg-per-pos $(NEG_PER_POS)

multi-seed-manip-v2:
	$(MN) --task manipulation --stage multi-seed --label-variant transition \
	      --multi-seeds 0,1,2,3,4 --max-offset $(MAX_OFFSET) \
	      --num-epochs $(EPOCHS) --neg-per-pos $(NEG_PER_POS)

multi-seed-decep-v2:
	$(DE) --task deception --stage multi-seed --label-variant transition \
	      --multi-seeds 0,1,2,3,4 --max-offset $(MAX_OFFSET) \
	      --num-epochs $(EPOCHS) --neg-per-pos $(NEG_PER_POS)

multi-seed-all-v2: multi-seed-sandbag-v2 multi-seed-manip-v2 multi-seed-decep-v2 \
                   pull-multi-seed-all \
                   multi-seed-aggregate

# Per-seed-per-subset pulls. modal volume get is finicky with bulk
# directory pulls; we fetch each results.json file explicitly. Safe to
# re-run.
.PHONY: pull-multi-seed-sandbag-v2 pull-multi-seed-manip-v2 pull-multi-seed-decep-v2 \
        pull-multi-seed-all

define PULL_SEEDS
	@for seed in 0 1 2 3 4; do \
	  for sub in all $(2); do \
	    mkdir -p results/$(1)/$(1)/results_transition/np10_mo10_ep200_seed$$seed/$$sub ; \
	    modal volume get --force $(1)-data \
	      $(1)/results_transition/np10_mo10_ep200_seed$$seed/$$sub/results.json \
	      results/$(1)/$(1)/results_transition/np10_mo10_ep200_seed$$seed/$$sub/results.json \
	      >/dev/null 2>&1 && echo "  pulled $(1) seed=$$seed $$sub" \
	      || echo "  SKIPPED $(1) seed=$$seed $$sub (not on volume)" ; \
	  done ; \
	done
endef

pull-multi-seed-sandbag-v2:
	$(call PULL_SEEDS,sandbag,sandbag_prompts)

pull-multi-seed-manip-v2:
	$(call PULL_SEEDS,manipulation,manipulation_prompts)

pull-multi-seed-decep-v2:
	$(call PULL_SEEDS,deception,deception_prompts)

pull-multi-seed-all: pull-multi-seed-sandbag-v2 pull-multi-seed-manip-v2 pull-multi-seed-decep-v2

# Local-only: needs numpy. Reads results/<exp>/.../np10_mo10_ep200_seed{0..4}/
# and prints + saves mean ± 95% CI per (layer, offset) cell, plus the
# best-layer-per-offset stability across seeds.
multi-seed-aggregate:
	python3 tools/multi_seed_aggregate.py --experiment sandbag      --label-variant transition --out results/sandbag/sandbag/multi_seed_aggregate_transition.json
	python3 tools/multi_seed_aggregate.py --experiment manipulation --label-variant transition --out results/manipulation/manipulation/multi_seed_aggregate_transition.json
	python3 tools/multi_seed_aggregate.py --experiment deception    --label-variant transition --out results/deception/deception/multi_seed_aggregate_transition.json

# ===== Rollout-level early-warning eval (deception only) =================
# Generates 240 fresh completions at sampling_seed=42, labels them with
# the v2 transition labeler, then scores every token with the trained
# L9 v2 deception probe. Computes "catch rate at k" — what fraction of
# committed-hack rollouts the probe flags ≥k tokens before the first
# lying token. Also reports the false-positive rate on honest rollouts.
#
# Modal time budget: ~15 min (generate) + ~10 min (label) + ~2 min
# (score+eval) ≈ 30 min.
.PHONY: rollout-decep rollout-generate-decep rollout-label-decep rollout-eval-decep

rollout-generate-decep:
	$(DE) --task deception --stage rollout-generate $(MODEL_FLAGS)

rollout-label-decep:
	$(DE) --task deception --stage rollout-label --label-variant transition $(LABEL_FLAGS)

rollout-eval-decep:
	$(DE) --task deception --stage rollout-eval --label-variant transition

# Convenience: end-to-end rollout pipeline (gen + label + eval), then pull.
rollout-decep:
	$(DE) --task deception --stage rollout-all --label-variant transition \
	      $(MODEL_FLAGS) $(LABEL_FLAGS)
	-$(MAKE) pull-decep-v2

# ----- profanity probe (legacy) -------------------------------------------
.PHONY: prof-all prof-model prof-label prof-probe prof-elicit prof-pull

prof-all:
	$(P) --stage all $(MODEL_FLAGS)

prof-model:
	$(P) --stage model $(MODEL_FLAGS)

prof-label:
	$(P) --stage label

prof-probe:
	$(P) --stage probe $(PROBE_FLAGS)

prof-elicit:
	$(P) --stage elicit

prof-pull:
	$(P) --stage download

# ----- QC: peek at samples.jsonl after labeling ---------------------------
.PHONY: qc-oneshot qc-explicit qc-syco qc-sandbag qc-sandbag-by-mode qc-sandbag-by-difficulty

qc-oneshot:
	@echo "=== substring_oneshot positive-rate by kind ==="
	@jq -s 'group_by(.prompt_kind) | map({kind: .[0].prompt_kind, n: length, n_pos: ([.[] | select(.has_positive)] | length)})' \
	    data/reward_hack/substring_oneshot/samples.jsonl

qc-explicit:
	@echo "=== substring_explicit positive-rate by kind ==="
	@jq -s 'group_by(.prompt_kind) | map({kind: .[0].prompt_kind, n: length, n_pos: ([.[] | select(.has_positive)] | length)})' \
	    data/reward_hack/substring_explicit/samples.jsonl

qc-syco:
	@echo "=== sycophancy positive-rate by kind ==="
	@jq -s 'group_by(.prompt_kind) | map({kind: .[0].prompt_kind, n: length, n_pos: ([.[] | select(.has_positive)] | length)})' \
	    data/reward_hack/sycophancy/samples.jsonl

qc-sandbag:
	@echo "=== sandbag positive-rate by kind ==="
	@jq -s 'group_by(.prompt_kind) | map({kind: .[0].prompt_kind, n: length, n_pos: ([.[] | select(.n_pos > 0)] | length)})' \
	    data/sandbag/sandbag/samples.jsonl

qc-sandbag-by-mode:
	@echo "=== sandbag positive-rate broken down by sandbag_mode (hack only) ==="
	@jq -s '[.[] | select(.prompt_kind=="hack")] | group_by(.sandbag_mode) | \
	        map({mode: .[0].sandbag_mode, n: length, n_pos: ([.[] | select(.n_pos > 0)] | length), \
	             mean_frac_pos: ((map(.frac_pos) | add) / length)})' \
	    data/sandbag/sandbag/samples.jsonl

qc-sandbag-by-difficulty:
	@echo "=== sandbag positive-rate broken down by difficulty x kind ==="
	@jq -s 'group_by(.difficulty + "/" + .prompt_kind) | \
	        map({bucket: (.[0].difficulty + "/" + .[0].prompt_kind), n: length, \
	             n_pos: ([.[] | select(.n_pos > 0)] | length), \
	             mean_frac_pos: ((map(.frac_pos) | add) / length)})' \
	    data/sandbag/sandbag/samples.jsonl

.PHONY: qc-sandbag-v2 qc-sandbag-by-mode-v2 qc-sandbag-compare

qc-sandbag-v2:
	@echo "=== sandbag v2 (transition) positive-rate by kind ==="
	@jq -s 'group_by(.prompt_kind) | map({kind: .[0].prompt_kind, n: length, n_pos: ([.[] | select(.n_pos > 0)] | length), mean_frac_pos: ((map(.frac_pos) | add) / length)})' \
	    data/sandbag/sandbag/samples_transition.jsonl

qc-sandbag-by-mode-v2:
	@echo "=== sandbag v2 (transition) positive-rate by sandbag_mode (hack only) ==="
	@jq -s '[.[] | select(.prompt_kind=="hack")] | group_by(.sandbag_mode) | \
	        map({mode: .[0].sandbag_mode, n: length, n_pos: ([.[] | select(.n_pos > 0)] | length), \
	             mean_frac_pos: ((map(.frac_pos) | add) / length)})' \
	    data/sandbag/sandbag/samples_transition.jsonl

qc-sandbag-compare:
	@echo "=== v1 (full) vs v2 (transition): per-token positive rate, hack-class ==="
	@printf "v1 (full)        :  "
	@jq -s '[.[] | select(.prompt_kind=="hack")] | {n: length, mean_frac_pos: ((map(.frac_pos) | add) / length)}' \
	    data/sandbag/sandbag/samples.jsonl
	@printf "v2 (transition)  :  "
	@jq -s '[.[] | select(.prompt_kind=="hack")] | {n: length, mean_frac_pos: ((map(.frac_pos) | add) / length)}' \
	    data/sandbag/sandbag/samples_transition.jsonl

.PHONY: qc-manip qc-manip-by-tactic qc-manip-v2 qc-manip-by-tactic-v2 qc-manip-compare

qc-manip:
	@echo "=== manipulation positive-rate by kind ==="
	@jq -s 'group_by(.prompt_kind) | map({kind: .[0].prompt_kind, n: length, n_pos: ([.[] | select(.n_pos > 0)] | length)})' \
	    data/manipulation/manipulation/samples.jsonl

qc-manip-by-tactic:
	@echo "=== manipulation positive-rate broken down by tactic (hack only) ==="
	@jq -s '[.[] | select(.prompt_kind=="hack")] | group_by(.tactic) | \
	        map({tactic: .[0].tactic, n: length, n_pos: ([.[] | select(.n_pos > 0)] | length), \
	             mean_frac_pos: ((map(.frac_pos) | add) / length)})' \
	    data/manipulation/manipulation/samples.jsonl

qc-manip-v2:
	@echo "=== manipulation v2 (transition) positive-rate by kind ==="
	@jq -s 'group_by(.prompt_kind) | map({kind: .[0].prompt_kind, n: length, n_pos: ([.[] | select(.n_pos > 0)] | length), mean_frac_pos: ((map(.frac_pos) | add) / length)})' \
	    data/manipulation/manipulation/samples_transition.jsonl

qc-manip-by-tactic-v2:
	@echo "=== manipulation v2 (transition) positive-rate by tactic (hack only) ==="
	@jq -s '[.[] | select(.prompt_kind=="hack")] | group_by(.tactic) | \
	        map({tactic: .[0].tactic, n: length, n_pos: ([.[] | select(.n_pos > 0)] | length), \
	             mean_frac_pos: ((map(.frac_pos) | add) / length)})' \
	    data/manipulation/manipulation/samples_transition.jsonl

qc-manip-compare:
	@echo "=== v1 (full) vs v2 (transition): per-token positive rate, hack-class ==="
	@printf "v1 (full)        :  "
	@jq -s '[.[] | select(.prompt_kind=="hack")] | {n: length, mean_frac_pos: ((map(.frac_pos) | add) / length)}' \
	    data/manipulation/manipulation/samples.jsonl
	@printf "v2 (transition)  :  "
	@jq -s '[.[] | select(.prompt_kind=="hack")] | {n: length, mean_frac_pos: ((map(.frac_pos) | add) / length)}' \
	    data/manipulation/manipulation/samples_transition.jsonl

.PHONY: qc-decep qc-decep-by-category qc-decep-v2 qc-decep-by-category-v2 qc-decep-compare

qc-decep:
	@echo "=== deception positive-rate by kind ==="
	@jq -s 'group_by(.prompt_kind) | map({kind: .[0].prompt_kind, n: length, n_pos: ([.[] | select(.n_pos > 0)] | length)})' \
	    data/deception/deception/samples.jsonl

qc-decep-by-category:
	@echo "=== deception positive-rate broken down by claim_category (hack only) ==="
	@jq -s '[.[] | select(.prompt_kind=="hack")] | group_by(.claim_category) | \
	        map({cat: .[0].claim_category, n: length, n_pos: ([.[] | select(.n_pos > 0)] | length), \
	             mean_frac_pos: ((map(.frac_pos) | add) / length)})' \
	    data/deception/deception/samples.jsonl

qc-decep-v2:
	@echo "=== deception v2 (transition) positive-rate by kind ==="
	@jq -s 'group_by(.prompt_kind) | map({kind: .[0].prompt_kind, n: length, n_pos: ([.[] | select(.n_pos > 0)] | length), mean_frac_pos: ((map(.frac_pos) | add) / length)})' \
	    data/deception/deception/samples_transition.jsonl

qc-decep-by-category-v2:
	@echo "=== deception v2 (transition) positive-rate by claim_category (hack only) ==="
	@jq -s '[.[] | select(.prompt_kind=="hack")] | group_by(.claim_category) | \
	        map({cat: .[0].claim_category, n: length, n_pos: ([.[] | select(.n_pos > 0)] | length), \
	             mean_frac_pos: ((map(.frac_pos) | add) / length)})' \
	    data/deception/deception/samples_transition.jsonl

qc-decep-compare:
	@echo "=== v1 (full) vs v2 (transition): per-token positive rate, hack-class ==="
	@printf "v1 (full)        :  "
	@jq -s '[.[] | select(.prompt_kind=="hack")] | {n: length, mean_frac_pos: ((map(.frac_pos) | add) / length)}' \
	    data/deception/deception/samples.jsonl
	@printf "v2 (transition)  :  "
	@jq -s '[.[] | select(.prompt_kind=="hack")] | {n: length, mean_frac_pos: ((map(.frac_pos) | add) / length)}' \
	    data/deception/deception/samples_transition.jsonl

# ----- help ---------------------------------------------------------------
.PHONY: help
help:
	@echo "Reward-hack probe (substring_oneshot / substring_explicit / sycophancy):"
	@echo "  make data-{oneshot,explicit,syco}     stage 1 only (generate corpus + activations)"
	@echo "  make label-{oneshot,explicit,syco}    stage 2 only (label tokens via OpenAI)"
	@echo "  make probe-{oneshot,explicit,syco}    stage 3 only (train + plot probes)"
	@echo "  make resume-{oneshot,explicit,syco}   label + probe (after data is good)"
	@echo "  make all-{oneshot,explicit,syco}      full pipeline"
	@echo "  make pull-{oneshot,explicit,syco}     download QC + results to ./data + ./results"
	@echo
	@echo "  make qc-{oneshot,explicit,syco}       jq summary of samples.jsonl"
	@echo
	@echo "  make data-all-rh / resume-all-rh / all-all-rh / pull-all-rh"
	@echo "    same operation across all 3 reward-hack tasks (sequential)"
	@echo
	@echo "Sandbagging probe (single task: sandbag):"
	@echo "  make data-sandbag / label-sandbag / probe-sandbag    v1 (full-span labels)"
	@echo "  make resume-sandbag                   label + probe (after data is good)"
	@echo "  make all-sandbag                      full pipeline"
	@echo "  make pull-sandbag                     download QC + results to ./data + ./results"
	@echo "  make qc-sandbag                       overall positive-rate by kind"
	@echo "  make qc-sandbag-by-mode               break down by sandbag_mode (hack only)"
	@echo "  make qc-sandbag-by-difficulty         break down by difficulty x kind"
	@echo
	@echo "Sandbagging v2 (transition-only labels — same corpus as v1):"
	@echo "  make label-sandbag-v2 / probe-sandbag-v2 / resume-sandbag-v2"
	@echo "  make all-sandbag-v2                   label + probe (skips model stage)"
	@echo "  make pull-sandbag-v2                  download v2 artifacts to ./data + ./results"
	@echo "  make qc-sandbag-v2                    v2 positive-rate by kind"
	@echo "  make qc-sandbag-by-mode-v2            v2 break down by mode"
	@echo "  make qc-sandbag-compare               v1 vs v2 hack-class positive-rate"
	@echo
	@echo "Manipulation probe (single task: manipulation):"
	@echo "  make data-manip / label-manip / probe-manip"
	@echo "  make resume-manip                     label + probe (after data is good)"
	@echo "  make all-manip                        full pipeline"
	@echo "  make pull-manip                       download QC + results to ./data + ./results"
	@echo "  make qc-manip                         overall positive-rate by kind"
	@echo "  make qc-manip-by-tactic               break down by tactic (hack only)"
	@echo
	@echo "Manipulation v2 (transition-only labels — same corpus as v1):"
	@echo "  make label-manip-v2 / probe-manip-v2 / resume-manip-v2"
	@echo "  make all-manip-v2                     label + probe (skips model stage)"
	@echo "  make pull-manip-v2                    download v2 artifacts to ./data + ./results"
	@echo "  make qc-manip-v2                      v2 positive-rate by kind"
	@echo "  make qc-manip-by-tactic-v2            v2 break down by tactic"
	@echo "  make qc-manip-compare                 v1 vs v2 hack-class positive-rate"
	@echo
	@echo "Deception probe (single task: deception):"
	@echo "  make data-decep / label-decep / probe-decep"
	@echo "  make resume-decep                     label + probe (after data is good)"
	@echo "  make all-decep                        full pipeline"
	@echo "  make pull-decep                       download QC + results to ./data + ./results"
	@echo "  make qc-decep                         overall positive-rate by kind"
	@echo "  make qc-decep-by-category             break down by claim_category (hack only)"
	@echo
	@echo "Deception v2 (transition-only labels — same corpus as v1):"
	@echo "  make label-decep-v2 / probe-decep-v2 / resume-decep-v2"
	@echo "  make all-decep-v2                     label + probe (skips model stage)"
	@echo "  make pull-decep-v2                    download v2 artifacts to ./data + ./results"
	@echo "  make qc-decep-v2                      v2 positive-rate by kind"
	@echo "  make qc-decep-by-category-v2          v2 break down by claim_category"
	@echo "  make qc-decep-compare                 v1 vs v2 hack-class positive-rate"
	@echo
	@echo "Follow-up probes (post-paper-v0 experiments):"
	@echo "  completion-level (mean-pool first N tokens, per-completion outcome):"
	@echo "    make completion-{sandbag,manip,decep}-v2   # uses transition labels"
	@echo "    make completion-manip-v1                   # v1 labels (manip headline)"
	@echo "  per-stratum (filter hack by category/tactic/mode):"
	@echo "    make per-stratum-{sandbag,manip,decep}-v2"
	@echo "  MLP-vs-linear sanity check (single layer, a few offsets):"
	@echo "    make mlp-sanity-{sandbag,manip,decep}-v2"
	@echo "  Probe direction similarity (local; reads downloaded probe_weights.pt):"
	@echo "    make probe-similarity                       # single L9 k=0 comparison"
	@echo "    make probe-similarity-sweep                 # full (layer, offset) grid"
	@echo "  meta targets:"
	@echo "    make followups-all                          # runs everything (compute + pulls + similarity)"
	@echo "    make followups-compute                      # compute-only (Modal jobs, no pulls)"
	@echo "    make followups-no-probes                    # like followups-all minus the v2 probe re-run"
	@echo "    make pull-followups                         # pull all 3 v2 trees + run probe-similarity"
	@echo
	@echo "Profanity probe (legacy):"
	@echo "  make prof-{all,model,label,probe,elicit,pull}"
	@echo
	@echo "Tunables (override on the command line, e.g. make label-oneshot WORKERS=2):"
	@echo "  LABELER_MODELS     = $(LABELER_MODELS)"
	@echo "  WORKERS            = $(WORKERS)"
	@echo "  PER_MODEL_ATTEMPTS = $(PER_MODEL_ATTEMPTS)"
	@echo "  MAX_TOTAL_WAIT     = $(MAX_TOTAL_WAIT)"
	@echo "  SAMPLES_PER_PROMPT = $(SAMPLES_PER_PROMPT)"
	@echo "  MAX_OFFSET         = $(MAX_OFFSET)"
	@echo "  EPOCHS             = $(EPOCHS)"
	@echo "  NEG_PER_POS        = $(NEG_PER_POS)"
	@echo "  RUN_NAME           = $(RUN_NAME)   (empty => auto-named subdir under results/)"
	@echo "  PREFIX_LENGTHS     = $(PREFIX_LENGTHS)   (completion-probe)"
	@echo "  MLP_LAYER          = $(MLP_LAYER)   (mlp-sanity-{sandbag,decep})"
	@echo "  MLP_LAYER_MANIP    = $(MLP_LAYER_MANIP)   (mlp-sanity-manip)"
	@echo "  MLP_OFFSETS        = $(MLP_OFFSETS)   (mlp-sanity)"
	@echo "  SEED               = $(SEED)   (rerun with --seed N for a fresh probe)"
