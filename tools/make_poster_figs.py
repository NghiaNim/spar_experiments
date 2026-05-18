"""Generate all poster figures.

Reads from results/* JSONs + probe_weights.pt, writes PNGs to
writeups/figures/poster/.

Run from repo root:
    python3 tools/make_poster_figs.py
"""
import json
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / 'writeups' / 'figures' / 'poster'
OUT.mkdir(parents=True, exist_ok=True)


def load(path):
    with open(ROOT / path) as f:
        return json.load(f)


# ============================================================
# fig0 — pipeline (unchanged from before, regenerated for completeness)
# ============================================================
def make_fig0():
    fig, ax = plt.subplots(figsize=(14, 4.5))
    ax.set_xlim(0, 14); ax.set_ylim(0, 5); ax.axis('off')

    def box(x, y, w, h, color, edge, text, fontsize=10, fw='normal'):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.05',
                                     facecolor=color, edgecolor=edge, linewidth=1.8))
        ax.text(x + w/2, y + h/2, text, ha='center', va='center',
                fontsize=fontsize, fontweight=fw)

    def arrow(x1, y1, x2, y2):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2),
                                       arrowstyle='->', mutation_scale=20,
                                       linewidth=2, color='#333'))

    box(0.2, 3.4, 2.4, 1.0, '#fce5cd', '#cc7a00',
        'Hack persona\n(e.g. "lie convincingly")', 9, 'bold')
    box(0.2, 1.2, 2.4, 1.0, '#d9ead3', '#38761d',
        'Honest persona\n(matched baseline)', 9, 'bold')
    box(3.4, 2.1, 2.4, 1.6, '#cfe2f3', '#1f6391',
        'Llama-3.2-1B\nabliterated\n(self-generates\ncompletions)', 10, 'bold')
    box(6.6, 2.8, 2.4, 1.6, '#d9d2e9', '#674ea7',
        'Per-token\nresidual-stream\nactivations\nat L = 0..16', 10, 'bold')
    box(6.6, 0.6, 2.4, 1.6, '#fff2cc', '#bf9000',
        'GPT-5.4-mini labels\ncommitment-span\ntokens\n(transition-only)', 10, 'bold')
    box(9.8, 2.1, 2.6, 1.6, '#f4cccc', '#990000',
        'Linear probe per\n(L, k): predict\nlabel_{t+k}\nfrom activation_t', 10, 'bold')
    box(12.8, 2.5, 1.0, 1.4, '#ead1dc', '#741b47',
        '17×11\nROC-AUC\n+ PR-AUC\nheatmap', 9, 'bold')
    box(12.8, 0.4, 1.0, 1.4, '#ead1dc', '#741b47',
        'Probe\ndirection\nW(L, k)', 9, 'bold')

    arrow(2.6, 3.9, 3.4, 3.0)
    arrow(2.6, 1.7, 3.4, 2.6)
    arrow(5.8, 3.2, 6.6, 3.5)
    arrow(5.8, 2.5, 6.6, 1.5)
    arrow(9.0, 3.5, 9.8, 3.0)
    arrow(9.0, 1.5, 9.8, 2.4)
    arrow(12.4, 3.0, 12.8, 3.0)
    arrow(12.4, 2.6, 12.8, 1.0)

    ax.text(7, 4.6, 'Three-stage probe pipeline', ha='center',
            fontsize=14, fontweight='bold')
    plt.savefig(OUT / 'fig0_pipeline.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('wrote', OUT / 'fig0_pipeline.png')


# ============================================================
# fig1 — heatmap panel (FIXED: tighter colorbar, band overlay, subtitle)
# ============================================================
def make_fig1():
    paths = {
        'Sandbag':      ('results/sandbag/sandbag/results_transition/np10_mo10_ep200/sandbag_prompts/results.json', 9,  'L9 wins 2/5 seeds'),
        'Manipulation': ('results/manipulation/manipulation/results_transition/np10_mo10_ep200/manipulation_prompts/results.json', 12, 'L12 wins 2/5 seeds'),
        'Deception':    ('results/deception/deception/results_transition/np10_mo10_ep200/deception_prompts/results.json', 9,  'L9 wins 4/5 seeds'),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), constrained_layout=True)
    for ax, (name, (path, headline_L, stability)) in zip(axes, paths.items()):
        d = load(path)
        layers = d['layers']
        offsets = d['offsets']
        auc = np.array(d['auc'])  # [L, k]

        # Tight colorbar — actual range is 0.50-0.75; use 0.50-0.78
        im = ax.imshow(auc, aspect='auto', origin='lower', cmap='viridis',
                       vmin=0.50, vmax=0.78, interpolation='nearest')

        ax.set_title(f'{name} (within-hack v2)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Lookahead offset k', fontsize=11)
        if name == 'Sandbag':
            ax.set_ylabel('Layer L', fontsize=11)
        ax.set_xticks(range(len(offsets))); ax.set_xticklabels(offsets, fontsize=9)
        ax.set_yticks(range(0, len(layers), 2)); ax.set_yticklabels([layers[i] for i in range(0, len(layers), 2)], fontsize=9)

        # Band overlay: rectangle around L7-L13 across all offsets
        L7_idx = layers.index(7); L13_idx = layers.index(13)
        band = Rectangle((-0.5, L7_idx - 0.5), len(offsets),
                         L13_idx - L7_idx + 1,
                         fill=False, edgecolor='red', linewidth=2.4, linestyle='-',
                         label='L7–L13 mid-stack band')
        ax.add_patch(band)

        # Star at headline cell
        L_idx = layers.index(headline_L)
        ax.scatter([0], [L_idx], marker='*', s=260, color='#ffd700',
                   edgecolor='black', linewidths=1.5, zorder=5)
        ax.annotate(f'L{headline_L}', xy=(0, L_idx), xytext=(1.3, L_idx + 0.4),
                    fontsize=10, fontweight='bold', color='black',
                    bbox=dict(boxstyle='round,pad=0.3', fc='#ffd700', ec='black', linewidth=0.8))

        # Per-panel subtitle showing seed stability
        ax.text(0.5, -0.27, stability, transform=ax.transAxes,
                ha='center', fontsize=10, fontweight='bold',
                color='#444', style='italic')

    cbar = fig.colorbar(im, ax=axes, shrink=0.85, location='right', pad=0.02)
    cbar.set_label('ROC-AUC', fontsize=12)
    fig.suptitle('Mid-stack band (L7–L13) carries the lookahead signal — but only deception\'s L9 is seed-robust',
                 fontsize=14, fontweight='bold')
    plt.savefig(OUT / 'fig1_heatmap_panel.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('wrote', OUT / 'fig1_heatmap_panel.png')


# ============================================================
# fig2 — lookahead (minor cleanup: keep as-is; substring noise is real)
# ============================================================
def make_fig2():
    lookahead = {
        'Substring (RH)':   ('results/reward_hack/substring_oneshot/results/np10_mo10_ep200/hack_prompts/results.json', 3,  '#888', '--'),
        'Sycophancy (RH)':  ('results/reward_hack/sycophancy/results/np10_mo10_ep200/hack_prompts/results.json',       9,  '#444', '--'),
        'Deception v2':     ('results/deception/deception/results_transition/np10_mo10_ep200/deception_prompts/results.json', 9,  '#1f77b4', '-'),
        'Sandbag v2':       ('results/sandbag/sandbag/results_transition/np10_mo10_ep200/sandbag_prompts/results.json',       9,  '#2ca02c', '-'),
        'Manipulation v2':  ('results/manipulation/manipulation/results_transition/np10_mo10_ep200/manipulation_prompts/results.json', 12, '#d62728', '-'),
    }

    fig, ax = plt.subplots(figsize=(7, 4.5), constrained_layout=True)
    for name, (path, L, color, style) in lookahead.items():
        d = load(path)
        layers = d['layers']; offsets = d['offsets']
        auc = np.array(d['auc'])
        roc_vs_k = auc[layers.index(L)]
        lw = 1.5 if 'RH' in name else 2.2
        ax.plot(offsets, roc_vs_k, marker='o', label=f'{name} (L{L})',
                linewidth=lw, linestyle=style, color=color, markersize=5)

    ax.axhline(0.5, color='gray', linestyle=':', linewidth=1)
    ax.set_xlabel('Lookahead offset k (tokens ahead of activation)', fontsize=11)
    ax.set_ylabel('ROC-AUC at headline layer', fontsize=11)
    ax.set_title('Lookahead horizon: probe ranking power decays with k',
                 fontsize=12, fontweight='bold')
    ax.set_xticks(offsets); ax.set_ylim(0.45, 1.02)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9, loc='lower left', framealpha=0.92)
    plt.savefig(OUT / 'fig2_lookahead.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('wrote', OUT / 'fig2_lookahead.png')


# ============================================================
# fig3 — rollout-eval (REDESIGNED: KDE on mean-score + catch-FPR difference)
# ============================================================
def make_fig3():
    d = load('results/deception/deception/rollout_eval_transition.json')
    threshold = d['threshold']

    hack_max  = np.array([r['max_score']  for r in d['hack']['rows']])
    honest_max = np.array([r['max_score'] for r in d['honest']['rows']])
    hack_mean   = np.array([r['mean_score']  for r in d['hack']['rows']])
    honest_mean = np.array([r['mean_score'] for r in d['honest']['rows']])

    # Committed-only catch
    hack_committed_max = np.array([r['max_score'] for r in d['hack']['rows'] if r['committed']])

    fig = plt.figure(figsize=(13, 5.4), constrained_layout=True)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.0])

    # ---- Panel A: KDE of MEAN-score per rollout ----
    axA = fig.add_subplot(gs[0, 0])
    # KDE with scipy not available — use histogram with density
    bins = np.linspace(0, 0.8, 25)
    axA.hist(honest_mean, bins=bins, alpha=0.6, label=f'Honest (mean={honest_mean.mean():.3f})',
             color='#1f77b4', edgecolor='black', linewidth=0.5, density=False)
    axA.hist(hack_mean, bins=bins, alpha=0.6, label=f'Hack (mean={hack_mean.mean():.3f})',
             color='#d62728', edgecolor='black', linewidth=0.5, density=False)
    axA.axvline(honest_mean.mean(), color='#1f77b4', linestyle='--', linewidth=2)
    axA.axvline(hack_mean.mean(), color='#d62728', linestyle='--', linewidth=2)
    axA.set_xlabel('Mean probe score per rollout', fontsize=11)
    axA.set_ylabel('# rollouts', fontsize=11)
    axA.set_title('A. Per-rollout mean score: honest > hack\n(probe fires MORE on honest)',
                  fontsize=11, fontweight='bold')
    axA.legend(fontsize=10, loc='upper right')
    axA.grid(True, alpha=0.3)

    # ---- Panel B: catch − FPR difference ----
    axB = fig.add_subplot(gs[0, 1])
    thresholds = np.linspace(0.5, 0.9999, 400)
    catch = np.array([(hack_committed_max > t).mean() for t in thresholds])
    fpr   = np.array([(honest_max         > t).mean() for t in thresholds])
    diff  = catch - fpr

    # Fill between curves
    axB.plot(thresholds, catch, label='Catch rate (committed hack)', color='#d62728', linewidth=3.0, linestyle='-')
    axB.plot(thresholds, fpr,   label='FPR (honest)',                color='#1f77b4', linewidth=3.0, linestyle=(0, (5, 2)))
    axB.fill_between(thresholds, catch, fpr, where=(catch > fpr),
                     color='#d62728', alpha=0.2, label='catch − FPR  (≈ 0 everywhere)')
    axB.fill_between(thresholds, catch, fpr, where=(catch <= fpr),
                     color='#1f77b4', alpha=0.2)
    axB.axvline(threshold, color='black', linestyle='--', linewidth=1.5, alpha=0.7,
                label=f'F1-tuned thresh = {threshold:.3f}')
    axB.set_xlabel('Threshold on max probe score', fontsize=11)
    axB.set_ylabel('Rate', fontsize=11)
    axB.set_title(f'B. catch − FPR ≈ 0 at every threshold\n(max |gap| = {np.abs(diff).max():.3f})',
                  fontsize=11, fontweight='bold')
    axB.legend(fontsize=9, loc='lower left', framealpha=0.92)
    axB.grid(True, alpha=0.3)
    axB.set_ylim(-0.02, 1.02)

    fig.suptitle('Deployment caveat: the within-hack probe does NOT separate hack from honest rollouts',
                 fontsize=14, fontweight='bold', color='#990000')
    plt.savefig(OUT / 'fig3_rollout_eval.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('wrote', OUT / 'fig3_rollout_eval.png')


# ============================================================
# fig4 — per-stratum WITH L0 baseline bars
# ============================================================
def make_fig4():
    # Recompute L0 PR-lift per stratum from the actual JSONs
    def stratum_data(behavior_root):
        results = []
        import os
        strata = sorted([d for d in os.listdir(ROOT / behavior_root)
                         if (ROOT / behavior_root / d / 'results.json').exists()])
        for s in strata:
            r = load(f'{behavior_root}/{s}/results.json')
            rows_k0 = [row for row in r['rows'] if row['offset'] == 0]
            L0 = next(row for row in rows_k0 if row['layer'] == 0)
            best = max(rows_k0, key=lambda row: row['pr_auc'])
            pos = L0['pos_rate_test']
            L0_lift   = L0['pr_auc']  - pos
            best_lift = best['pr_auc'] - pos
            results.append((s, best_lift, L0_lift, pos))
        # Sort by best_lift desc
        return sorted(results, key=lambda x: -x[1])

    behaviors = {
        'Sandbag (modes)':         ('results/sandbag/sandbag/results_strata_transition', '#2ca02c', 0.217),
        'Manipulation (tactics)':  ('results/manipulation/manipulation/results_strata_transition', '#d62728', 0.118),
        'Deception (categories)':  ('results/deception/deception/results_strata_transition', '#1f77b4', 0.203),
    }

    fig, axes = plt.subplots(1, 3, figsize=(15, 4.4), constrained_layout=True)
    for ax, (label, (root, color, headline)) in zip(axes, behaviors.items()):
        data = stratum_data(root)
        names = [d[0] for d in data]
        best_lifts = [d[1] for d in data]
        L0_lifts = [max(d[2], 0) for d in data]  # clip negative L0 lifts at 0 for visual sanity

        y = np.arange(len(names))
        h = 0.4
        ax.barh(y - h/2, best_lifts, height=h, color=color, alpha=0.9,
                edgecolor='black', linewidth=0.5, label='best-layer lift')
        ax.barh(y + h/2, L0_lifts, height=h, color=color, alpha=0.35,
                edgecolor='black', linewidth=0.5, hatch='//', label='L0 (embedding) lift')

        ax.set_yticks(y); ax.set_yticklabels(names, fontsize=10)
        ax.invert_yaxis()
        ax.set_xlabel('PR-AUC lift over baseline @ k=0', fontsize=10)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.set_xlim(-0.02, 0.62)
        ax.axvline(headline, color='black', linestyle='--', linewidth=1.4, alpha=0.7,
                   label=f'full-corpus +{headline:.3f}')
        ax.grid(True, alpha=0.3, axis='x')
        ax.legend(fontsize=8, loc='lower right', framealpha=0.92)

        # Annotate
        for i, (b, l0) in enumerate(zip(best_lifts, L0_lifts)):
            ax.text(b + 0.008, i - h/2, f'+{b:.2f}', va='center', fontsize=8)
            if l0 > 0.005:
                ax.text(l0 + 0.008, i + h/2, f'+{l0:.2f}', va='center', fontsize=8, alpha=0.7)

    fig.suptitle('Per-stratum PR-AUC lift: full bar = best-layer, hatched = L0 (embedding). When the two bars overlap → lexical detector.',
                 fontsize=13, fontweight='bold')
    plt.savefig(OUT / 'fig4_strata.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('wrote', OUT / 'fig4_strata.png')


# ============================================================
# fig5 — cosine similarity 3x3 (kept compact for inset use)
# ============================================================
def make_fig5():
    sim = load('results/probe_similarity_L9_k0.json')
    mat = np.array(sim['cosine_matrix'])
    labels = ['Sandbag', 'Manip.', 'Deception']

    fig, ax = plt.subplots(figsize=(4.6, 4.0), constrained_layout=True)
    # Mask the diagonal (set to NaN), so colorbar is set by off-diagonal scale
    mat_off = mat.copy()
    np.fill_diagonal(mat_off, np.nan)

    cmap = plt.cm.RdBu_r.copy()
    cmap.set_bad('#dddddd')
    im = ax.imshow(mat_off, vmin=-0.05, vmax=0.05, cmap=cmap, interpolation='nearest')

    for i in range(3):
        for j in range(3):
            val = mat[i, j]
            if i == j:
                ax.text(j, i, 'self', ha='center', va='center', fontsize=11,
                        color='#888', fontweight='bold')
            else:
                ax.text(j, i, f'{val:+.3f}', ha='center', va='center', fontsize=13,
                        color='black', fontweight='bold')

    ax.set_xticks(range(3)); ax.set_xticklabels(labels, fontsize=11)
    ax.set_yticks(range(3)); ax.set_yticklabels(labels, fontsize=11)
    ax.set_title('Probe-direction cosine sim @ L9, k=0\n(noise floor ±0.022 for H=2048)',
                 fontsize=11, fontweight='bold')
    cbar = fig.colorbar(im, ax=ax, shrink=0.7)
    cbar.set_label('cos similarity', fontsize=10)
    plt.savefig(OUT / 'fig5_cosine_sim.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('wrote', OUT / 'fig5_cosine_sim.png')


# ============================================================
# NEW fig6 — multi-seed CIs + best-layer stability
# ============================================================
def make_fig6():
    aggregates = {
        'Sandbag':      ('results/sandbag/sandbag/multi_seed_aggregate_transition.json',           9,  '#2ca02c'),
        'Manipulation': ('results/manipulation/manipulation/multi_seed_aggregate_transition.json', 12, '#d62728'),
        'Deception':    ('results/deception/deception/multi_seed_aggregate_transition.json',       9,  '#1f77b4'),
    }

    fig, axes = plt.subplots(2, 3, figsize=(14, 5.6),
                              gridspec_kw={'height_ratios': [3.2, 0.8]},
                              constrained_layout=True)
    for col, (name, (path, headline_L, color)) in enumerate(aggregates.items()):
        d = load(path)
        offsets = d['offsets']
        rows = d['rows']

        # ROC mean ± CI at headline layer
        roc_mean = []
        roc_lo = []
        roc_hi = []
        for k in offsets:
            row = next(r for r in rows if r['layer'] == headline_L and r['offset'] == k)
            roc_mean.append(row['roc_auc_mean'])
            roc_lo.append(row['roc_auc_lo'])
            roc_hi.append(row['roc_auc_hi'])
        roc_mean = np.array(roc_mean)
        roc_lo = np.array(roc_lo)
        roc_hi = np.array(roc_hi)

        ax = axes[0, col]
        ax.plot(offsets, roc_mean, marker='o', color=color, linewidth=2.2,
                markersize=6, label=f'L{headline_L} mean ROC')
        ax.fill_between(offsets, roc_lo, roc_hi, color=color, alpha=0.25,
                        label='95% bootstrap CI')
        ax.axhline(0.5, color='gray', linestyle=':', linewidth=1)
        ax.set_xlabel('Lookahead offset k', fontsize=10)
        if col == 0:
            ax.set_ylabel(f'ROC-AUC at headline layer', fontsize=10)
        ax.set_title(f'{name} (L{headline_L}, n=5 seeds)', fontsize=12, fontweight='bold')
        ax.set_ylim(0.45, 0.85)
        ax.set_xticks(offsets)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, loc='lower left')

        # Bottom panel: best-layer stability across k
        ax2 = axes[1, col]
        n_seeds = d['n_seeds_loaded']
        fractions = []
        for k in offsets:
            pob = next(p for p in d['per_offset_best'] if p['offset'] == k)
            dist = pob['best_layer_dist_across_seeds']
            count_at_headline = dist.get(str(headline_L), 0)
            fractions.append(count_at_headline / n_seeds)
        bars = ax2.bar(offsets, fractions, color=color, alpha=0.85,
                       edgecolor='black', linewidth=0.5)
        ax2.set_ylim(0, 1.05)
        ax2.set_xlabel('Lookahead offset k', fontsize=10)
        if col == 0:
            ax2.set_ylabel(f'frac. seeds\nbest = L{headline_L}', fontsize=9)
        ax2.set_xticks(offsets)
        ax2.set_yticks([0, 0.5, 1.0])
        ax2.set_yticklabels(['0', '2.5/5', '5/5'])
        ax2.axhline(0.5, color='gray', linestyle=':', linewidth=0.8)
        ax2.grid(True, alpha=0.3, axis='y')
        # annotate counts on bars
        for k, f in zip(offsets, fractions):
            ax2.text(k, f + 0.03, f'{int(f*n_seeds)}/{n_seeds}',
                     ha='center', va='bottom', fontsize=7.5)

    fig.suptitle('Multi-seed (5 seeds × 3 behaviors): the band is robust; the single-cell winner is shakier',
                 fontsize=13, fontweight='bold')
    plt.savefig(OUT / 'fig6_multiseed.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('wrote', OUT / 'fig6_multiseed.png')


# ============================================================
# NEW fig7 — held-out category eval (deception)
# ============================================================
def make_fig7():
    d = load('results/deception/deception/results_held_out_transition/summary.json')
    rows = sorted(d['rows'], key=lambda r: -r['L9_k0_roc'])

    names = [r['held_out'] for r in rows]
    L9_roc = [r['L9_k0_roc'] for r in rows]
    best_roc = [r['best_roc_k0'] for r in rows]
    best_L = [r['best_roc_k0_L'] for r in rows]

    within_category_v2_roc = 0.751  # from §3.3 deception
    mean_L9 = np.mean(L9_roc)

    fig, ax = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    y = np.arange(len(names))
    h = 0.4
    bars1 = ax.barh(y - h/2, L9_roc, height=h, color='#1f77b4', alpha=0.9,
                    edgecolor='black', linewidth=0.5, label='L9 ROC (held out)')
    bars2 = ax.barh(y + h/2, best_roc, height=h, color='#1f77b4', alpha=0.40,
                    edgecolor='black', linewidth=0.5, hatch='//',
                    label='best-anywhere ROC')

    ax.set_yticks(y); ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlabel('ROC-AUC @ k=0  (trained on 4 / tested on the held-out 5th)', fontsize=10)
    ax.set_title('Held-out category eval: L9 generalizes across deception types\n(mean L9 ROC = 0.713 across 5 LOO conditions)',
                 fontsize=12, fontweight='bold')
    ax.set_xlim(0.5, 0.88)
    ax.axvline(within_category_v2_roc, color='black', linestyle='--', linewidth=1.4,
               label=f'within-category v2 ROC = {within_category_v2_roc:.3f}')
    ax.axvline(mean_L9, color='#1f77b4', linestyle=':', linewidth=2,
               label=f'mean L9 (held-out) = {mean_L9:.3f}')
    ax.grid(True, alpha=0.3, axis='x')
    ax.legend(fontsize=9, loc='lower right', framealpha=0.92)

    # Annotate values
    for i, (r, b, L) in enumerate(zip(L9_roc, best_roc, best_L)):
        ax.text(r + 0.005, i - h/2, f'{r:.3f}', va='center', fontsize=9)
        ax.text(b + 0.005, i + h/2, f'{b:.3f} (L{L})', va='center', fontsize=8, alpha=0.7)

    plt.savefig(OUT / 'fig7_held_out.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('wrote', OUT / 'fig7_held_out.png')


# ============================================================
# NEW fig8 — v1 vs v2 labeling diagnostic
# ============================================================
def make_fig8():
    """v1 vs v2 PR-AUC lift, grouped by behavior, with arrows showing direction."""
    behaviors = ['Sandbag', 'Deception', 'Manipulation']
    v1_lift =  [0.151, 0.147, 0.226]
    v2_lift =  [0.217, 0.203, 0.118]
    colors   = ['#2ca02c', '#1f77b4', '#d62728']

    fig, ax = plt.subplots(figsize=(7.5, 4.4), constrained_layout=True)
    x = np.arange(len(behaviors))
    w = 0.34

    bars1 = ax.bar(x - w/2, v1_lift, width=w, color=colors, alpha=0.45,
                   edgecolor='black', linewidth=0.7, hatch='//',
                   label='v1 (full-span labels)')
    bars2 = ax.bar(x + w/2, v2_lift, width=w, color=colors, alpha=0.95,
                   edgecolor='black', linewidth=0.7,
                   label='v2 (transition-only labels)')

    # Arrows v1 → v2
    for i, (a, b) in enumerate(zip(v1_lift, v2_lift)):
        color_arrow = '#0a8a00' if b > a else '#990000'
        ax.annotate('', xy=(i + w/2, b + 0.005), xytext=(i - w/2, a + 0.005),
                    arrowprops=dict(arrowstyle='->', color=color_arrow,
                                    linewidth=2.5))
        # Δ label
        delta = b - a
        ax.text(i, max(a, b) + 0.030, f'Δ = {delta:+.3f}',
                ha='center', fontsize=11, fontweight='bold',
                color=color_arrow)
        # Behavior-shape label
        shape = 'fleeting commitment' if b > a else 'sustained register'
        ax.text(i, -0.025, shape, ha='center', fontsize=9,
                fontstyle='italic', color=color_arrow)

    ax.set_xticks(x); ax.set_xticklabels(behaviors, fontsize=12, fontweight='bold')
    ax.set_ylabel('PR-AUC lift over baseline (within-hack, k=0)', fontsize=11)
    ax.set_title('Labeling target × behavior shape:\ntransition labels help fleeting commitments, hurt sustained registers',
                 fontsize=12, fontweight='bold')
    ax.set_ylim(-0.05, 0.32)
    ax.axhline(0, color='black', linewidth=0.8)
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(True, alpha=0.3, axis='y')

    plt.savefig(OUT / 'fig8_v1_v2.png', dpi=200, bbox_inches='tight')
    plt.close()
    print('wrote', OUT / 'fig8_v1_v2.png')


if __name__ == '__main__':
    make_fig0()
    make_fig1()
    make_fig2()
    make_fig3()
    make_fig4()
    make_fig5()
    make_fig6()
    make_fig7()
    make_fig8()
    print('\nAll figures written to', OUT)
