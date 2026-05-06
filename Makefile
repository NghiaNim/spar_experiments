# Shortcuts for the probe pipeline. Run `make help` for the menu.
#
# Two experiments:
#   - reward-hack probe (3 task variants: oneshot / explicit / syco)
#   - profanity probe (legacy, prefixed `prof-`)
#
# The most common workflow once data is already on the Modal volume:
#   make resume-oneshot     # re-label and re-probe
#
# Override any tunable on the command line, e.g.
#   make label-oneshot WORKERS=2 LABELER_MODELS=gpt-5.4
#   make probe-syco EPOCHS=400 NEG_PER_POS=5

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
.PHONY: qc-oneshot qc-explicit qc-syco

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
