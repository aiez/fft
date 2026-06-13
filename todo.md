# fft todo

## QuickEst-style regression list (idea, measured, not yet in fft.py)

Context: explored whether FFTs do *regression* (continuous target, here
d2h). They do -- QuickEst (Hertwig, Hoffrage & Martignon 1999), the
regression sibling of classification FFTs (Martignon'08, Woike'17). All
the FFT-classification papers compare vs CART/logistic but the task is
binary; "regression" there is only CART's name. QuickEst is the real
continuous one: leaf stores a mean, lexicographic one-reason scan.

Key realisation: it is NOT a tree. It is a flat decision list. The tree
shape only matters if a leaf value depends on the path; with a fixed
cue order it collapses to a list + two loops. Conditioning the tags
(below) is just a shrinking row-set in the learn loop, not branching.

### Final pseudocode (plain QuickEst, conditional tags)

    # LEARN
    make_list(rows):
        features = every "value-in-range" test, 1+ per column
        sort features by support, smallest first   # support = #rows that PASS f
        live = rows
        for f in features:                          # walk in that order
            exiting = live rows that FAIL f         # rows that stop here
            f.tag   = average target of exiting     # CONDITIONAL: only these
            live    = live rows that PASS f          # survivors carry forward
        catchall = average target of live            # passed everything
        return features, catchall

    # GUESS (one-reason: stop at first failed test)
    guess(features, row):
        for f in features:
            if row FAILS f: return f.tag
        return catchall

Frugal: smallest-support-first => first test is one almost no row
passes => most rows fail it and exit at line 1 with the common-case
average. Only unusual rows walk deep; the rare all-pass rows hit
catchall. Most guesses cost one question.

### Three knobs (each ~0.03-0.04 RMSE on 5 optimiz sets)

    knob        QuickEst (simple)        ifan (accurate, = fft.py now)
    --------    ----------------------   ----------------------------
    1 order     by support               by variance-reduction score
    2 direction fixed (exit fail side)   fan both sides, keep best on train
    3 tag       marginal (all rows)      conditional (survivors only)

QuickEst = all knobs left; fft.py's ifan = all knobs right.

### Ranges: bottom-up fuse (supervised discretizer, orthogonal to knobs)

Start 10 bins/col; merge adjacent bins while merged sd <= weighted-avg
of parts' sd. Recurse to fixpoint. Splits survive only where y shifts;
yields ~2-6 ranges/col. Clean ~8 lines using existing merge/sd/n_.

    fuse(bins):
        repeat until no change:
            for adjacent a,b:
                c = merge(a,b)
                if sd(c) <= (a.n*sd(a)+b.n*sd(b))/c.n: replace a,b with c
        return bins

### Verdict (measured, 30 + 5 datasets, 5 seeds, 50/50)

- As a PREDICTOR (RMSE): ifan (current fft.py) beats every QuickEst/rake
  /list variant on all sets; loses to gradient boosting (>=16 trees) on
  fine-grained config data (e.g. SS-N) by the d+1-leaf vocabulary ceiling.
- As an OPTIMIZER (rank test by tree, look up top-5, return best): tree
  adds +23 win pts over random-5 (76 vs 53); coarse ranking is enough,
  so the d+1 ceiling stops mattering. This is eval.py's bob/acquire game.
- QuickEst list: simplest, tree-free, citeable; a few RMSE pts behind
  ifan because of marginal tags + support order + fixed direction.

### Next (optional, unbuilt)

- [ ] Feed fused cut-points into ifan's `cuts` (knob 1, sharper
      thresholds) -- only combination not yet measured.
- [ ] Ship the flat list as quicklist.py sibling for teaching contrast
      (same FFT family; one design choice decides tree vs list).
