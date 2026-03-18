"""Color schemes optimized for analytical clarity, not aesthetics.

Claude picks color schemes per user decision. These are defaults
for mathematical visualization where contrast and distinguishability matter.
"""

# Analytical palette for line plots
ANALYTICAL_PALETTE = {
    "primary": "#1f77b4",       # Blue -- main data
    "secondary": "#ff7f0e",     # Orange -- comparison/overlay
    "zero_line": "#d62728",     # Red -- zero crossings, critical values
    "grid": "#cccccc",          # Light gray -- background grid
    "annotation": "#2ca02c",    # Green -- annotations, highlights
    "background": "#ffffff",    # White -- clean background
}

# Matplotlib style defaults for analytical plots
MATPLOTLIB_DEFAULTS = {
    "figure.figsize": (12, 6),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "lines.linewidth": 1.0,
    "font.size": 10,
}
