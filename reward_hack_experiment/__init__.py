"""Reward-hacking probe experiment pipeline.

Sister project to ``probe_experiment``. Same probe machinery, different
target behavior: instead of predicting upcoming profane tokens we predict
upcoming **reward-hacking** tokens. Two tasks are supported:

  - ``substring``  — specification gaming via a keyword scanner.
  - ``sycophancy`` — agreement with a verifiably-wrong student claim.

See ``reward_hack_experiment/seed_prompts.py`` for the per-task personas
and the user-prompt corpora.
"""
