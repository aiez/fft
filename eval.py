#!/usr/bin/env python3 -B
"""eval.py: score each fft tree on perf + fairness goals.

Build trees on -t trainN sampled rows (seeded), eval on the rest.
Per tree report: pd, pf, prec, ard, aaod (one each) + spd per
protected (~) col. pos label from fairnez/labels.py (else minority).

Uses fft.py as-is (grows/predict/Data/csv); y = 1.0 if pos else 0.0,
leaf mean thresholded at 0.5 = predicted class.

Options:
 -f file     data file  ($DOOT/fairnez/adult.csv)
 -t trainN   train rows  (200)
 -P pos      positive label  (auto)
"""
import sys, os, random, importlib.util
import fft
from fft import Data, grows, predict, csv, of, o

EPS = 1e-32

#-- confusion / metrics -----------------------------------------
def confused(pairs):                 # pairs=[(want,got),...] one-vs-rest
  n, labels, N = {}, set(), len(pairs)
  for wg in pairs:
    n[wg] = n.get(wg, 0) + 1
    labels |= set(wg)
  out = {}
  for k in labels:
    tp = n.get((k, k), 0)
    fn = sum(c for (w, g), c in n.items() if w == k and g != k)
    fp = sum(c for (w, g), c in n.items() if w != k and g == k)
    tn = N - tp - fn - fp
    pd, pf = tp/(tp+fn+EPS), fp/(fp+tn+EPS)
    prec   = tp/(tp+fp+EPS)
    out[k] = dict(tp=tp, fp=fp, fn=fn, tn=tn, pd=pd, pf=pf, prec=prec,
                  acc=(tp+tn)/(N+EPS))
  return out

def rate(m): return (m["tp"]+m["fp"]) / (m["tp"]+m["fp"]+m["fn"]+m["tn"]+EPS)
def spd(a, b, pos): return abs(rate(a[pos]) - rate(b[pos]))

#-- protected grouping ------------------------------------------
def groups(rows, at, isnum):         # -> (g1rows,g1, g2rows,g2)
  if isnum:                          # median (mean) split
    vs  = [r[at] for r in rows if r[at] != "?"]
    med = sum(vs)/len(vs) if vs else 0
    return ([r for r in rows if r[at] != "?" and r[at] <= med], "lo",
            [r for r in rows if r[at] != "?" and r[at] >  med], "hi")
  freq = {}                          # top-2 most-frequent syms
  for r in rows:
    if r[at] != "?": freq[r[at]] = freq.get(r[at], 0) + 1
  top = sorted(freq, key=freq.get, reverse=True)[:2]
  if len(top) < 2: return ([], None, [], None)
  g1, g2 = top
  return ([r for r in rows if r[at] == g1], g1,
          [r for r in rows if r[at] == g2], g2)

#-- pos label ---------------------------------------------------
def lookupPos(path):                 # fairnez/labels.py POSITIVE map
  lab = os.path.join(os.path.dirname(os.path.abspath(path)), "labels.py")
  if not os.path.exists(lab): return None
  spec = importlib.util.spec_from_file_location("labels", lab)
  mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
  return getattr(mod, "POSITIVE", {}).get(os.path.basename(path))

#-- print -------------------------------------------------------
def print2d(rows, sep="  "):
  rows = [[str(x) for x in r] for r in rows]
  ws = [max(len(r[c]) for r in rows if c < len(r))
        for c in range(max(len(r) for r in rows))]
  for r in rows:
    print(sep.join(r[c].rjust(ws[c]) for c in range(len(r))))

def scoreTree(t, test, klass, prot, names, lab, got):  # -> [pd,pf,prec,*spds]
  m = confused([(lab(r[klass]), got(t, r)) for r in test])["pos"]
  spds = []
  for at, s in prot:
    isnum = names[at][0].isupper()
    g1, n1, g2, n2 = groups(test, at, isnum)
    if not g1 or not g2: spds.append(0); continue
    a = confused([(lab(r[klass]), got(t, r)) for r in g1])
    b = confused([(lab(r[klass]), got(t, r)) for r in g2])
    if "pos" not in a or "pos" not in b: spds.append(0); continue
    spds.append(spd(a, b, "pos"))
  return [m["pd"], m["pf"], m["prec"]] + spds

def d2h(scores, names2):             # disty per score row vs the board
  board = Data([names2] + scores)
  return [fft.disty(board, r) for r in scores]

#-- shared setup ------------------------------------------------
def prep(file, pos):
  rows = list(csv(file))
  names, body = rows[0], rows[1:]
  klass = next(at for at, s in enumerate(names) if s[-1] == "!")
  prot  = [(at, s) for at, s in enumerate(names) if s.endswith("~")]
  if pos in (None, "", "auto"): pos = lookupPos(file)
  if pos is None:                    # minority label
    cnt = {}
    for r in body: cnt[r[klass]] = cnt.get(r[klass], 0) + 1
    pos = min(cnt, key=cnt.get)
  names2 = ["idX", "biasX", "Pd+", "Pf-", "Prec+"] \
           + ["Spd_" + s.rstrip("~") + "-" for _, s in prot]
  ctx = o(names=names, body=body, klass=klass, prot=prot, pos=pos,
          names2=names2,
          y   =lambda r: 1.0 if r[klass] == pos else 0.0,
          lab =lambda v: "pos" if v == pos else "neg",
          got =lambda t, r: "pos" if predict(t, r) >= 0.5 else "neg")
  return ctx

def score1(c, t, test):              # [idless] score row goals only
  return scoreTree(t, test, c.klass, c.prot, c.names, c.lab, c.got)

#-- main: full table --------------------------------------------
def main(file, trainN, pos, repeats):
  c = prep(file, pos)
  scores = []
  for rep in range(1, repeats + 1):
    shuf = random.sample(c.body, len(c.body))
    train, test = shuf[:trainN], shuf[trainN:]
    data = Data([c.names] + train)
    for bias, t in grows(data.rows, c.y, data):
      scores.append([len(scores)+1, "%d:%s" % (rep, bias or "-")]
                    + score1(c, t, test))
  ds     = d2h(scores, c.names2)
  ranked = sorted(zip(ds, scores), key=lambda x: x[0])
  header = ["disty"] + [s[:-1] if s[-1] == "X" else s for s in c.names2]
  table  = [header] + [["%.3f" % d, r[0], r[1]]
                       + ["%.2f" % v for v in r[2:]] for d, r in ranked]
  print(os.path.basename(file),
        " pos=%s train=%d repeats=%d rows=%d  (sorted by disty)" %
        (c.pos, trainN, repeats, len(scores)))
  print2d(table)

#-- compare: full grid vs successive halving --------------------
# cands = pool of trees (built once). full = score every cand on the
# whole test pool. SHA = score on `start` test rows, keep top half by
# d2h, double test rows, repeat until 1 survivor. cost = predictions
# = sum(survivors * testrows). winner judged by its FULL-pool d2h.
def buildCands(c, trainN, repeats):
  pool = random.sample(c.body, len(c.body))
  cut  = int(0.5 * len(pool))
  test, trainpool = pool[:cut], pool[cut:]   # fixed test pool
  cands = []
  for _ in range(repeats):
    tr   = random.sample(trainpool, min(trainN, len(trainpool)))
    data = Data([c.names] + tr)
    cands += [t for _, t in grows(data.rows, c.y, data)]
  return cands, test

def evalPool(c, cands, test):        # d2h of every cand on `test`
  scores = [[i, ""] + score1(c, t, test) for i, t in enumerate(cands)]
  return d2h(scores, c.names2)       # aligned to cands order

def full(c, cands, test):
  ds   = evalPool(c, cands, test)
  best = min(range(len(cands)), key=lambda i: ds[i])
  cost = len(cands) * len(test)
  return best, cost

def sha(c, cands, test, start=50):
  alive = list(range(len(cands)))
  n, cost = start, 0
  while len(alive) > 1 and n <= len(test):
    sub  = test[:n]
    ds   = evalPool(c, [cands[j] for j in alive], sub)
    cost += len(alive) * len(sub)
    order = sorted(range(len(alive)), key=lambda k: ds[k])
    keep  = max(1, len(alive) // 2)
    alive = [alive[k] for k in order[:keep]]
    n    *= 2
  if len(alive) > 1:                 # tie-break survivors on full test
    ds    = evalPool(c, [cands[j] for j in alive], test)
    cost += len(alive) * len(test)
    alive = [alive[min(range(len(alive)), key=lambda k: ds[k])]]
  return alive[0], cost

def race(c, cands, test, start=50, delta=0.1):
  # Hoeffding-style race: keep any cand within a noise margin of the
  # current best; margin = sqrt(ln(2/delta)/2n) shrinks as test grows.
  # losers die fast (cheap); the true best is never culled on noise.
  from math import log, sqrt
  alive, n, cost, ds = list(range(len(cands))), start, 0, [0]
  while len(alive) > 1 and n < len(test):
    ds     = evalPool(c, [cands[j] for j in alive], test[:n])
    cost  += len(alive) * n
    best   = min(ds)
    margin = sqrt(log(2/delta) / (2*n))
    keep   = [k for k, d in enumerate(ds) if d <= best + margin]
    alive  = [alive[k] for k in keep]
    ds     = [ds[k] for k in keep]              # decide on biggest rung seen
    n     *= 2
  return alive[min(range(len(alive)), key=lambda k: ds[k])], cost

#-- final: race on train, report best tree's goals on test ------
# split body 50:50. learn `repeats`x16 trees on 100-row train samples.
# race over ALL training rows -> 1 best tree. score it on TEST. one row.
# every goal cell = fit/train/test (integer percents). fit = the 100
# rows the winning tree actually grew on; train = all train rows;
# test = held-out. fit>>test gap = overfitting.
FINALHEAD = ["data", "pos", "Pd+", "Pf-", "Prec+", "SpdMean-", "spds"]

def finalRun(file, trainN, pos, repeats, start, asCsv=False):
  c = prep(file, pos)
  shuf = random.sample(c.body, len(c.body))
  cut  = len(shuf) // 2
  train, test = shuf[:cut], shuf[cut:]
  cands, fits = [], []                          # 160 trees + their fit rows
  for _ in range(repeats):
    tr = random.sample(train, min(trainN, len(train)))
    d  = Data([c.names] + tr)
    for _, t in grows(d, c.y, d): cands.append(t); fits.append(tr)
  best, _ = race(c, cands, train, start)        # race on ALL train data
  gfi = score1(c, cands[best], fits[best])      # goals on FIT rows (seen)
  gtr = score1(c, cands[best], train)           # goals on all TRAIN
  gte = score1(c, cands[best], test)            # goals on held-out TEST
  pct = lambda v: round(100*v)
  ftt = lambda f, a, b: "%d/%d/%d" % (pct(f), pct(a), pct(b))
  sfi, str_, ste = gfi[3:], gtr[3:], gte[3:]
  mean   = lambda xs: sum(xs)/len(xs) if xs else 0
  detail = ";".join("%s=%s" % (s.rstrip("~"), ftt(f, a, b))
                    for (_, s), f, a, b in zip(c.prot, sfi, str_, ste))
  vals = [os.path.basename(file), c.pos,
          ftt(gfi[0], gtr[0], gte[0]), ftt(gfi[1], gtr[1], gte[1]),
          ftt(gfi[2], gtr[2], gte[2]),
          ftt(mean(sfi), mean(str_), mean(ste)), detail]
  if asCsv: print(",".join(str(x) for x in vals))
  else:     print2d([FINALHEAD, vals])

CSVHEAD = ["data", "test", "cands", "full_s", "sha_s", "race_s",
           "full_rank", "sha_rank", "race_rank",
           "full_d2h", "sha_d2h", "race_d2h",
           "full_cost", "sha_cost", "race_cost"]

def compare(file, trainN, pos, repeats, start, asCsv=False):
  import time
  c = prep(file, pos)
  cands, test = buildCands(c, trainN, repeats)
  fullds = evalPool(c, cands, test)            # ground-truth ranking
  order  = sorted(range(len(cands)), key=lambda i: fullds[i])
  rank   = {i: k for k, i in enumerate(order)}

  res = {}                                      # method -> (winner,cost,time)
  for name, fn in (("full", lambda: full(c, cands, test)),
                   ("sha",  lambda: sha(c, cands, test, start)),
                   ("race", lambda: race(c, cands, test, start))):
    t0 = time.perf_counter()
    win, cost = fn()
    res[name] = (win, cost, time.perf_counter() - t0)

  if asCsv:
    nm = os.path.basename(file)
    print(",".join(str(x) for x in [
      nm, len(test), len(cands),
      *["%.3f" % res[m][2] for m in ("full", "sha", "race")],
      *[rank[res[m][0]] for m in ("full", "sha", "race")],
      *["%.3f" % fullds[res[m][0]] for m in ("full", "sha", "race")],
      *[res[m][1] for m in ("full", "sha", "race")]]))
    return
  out = [["method", "winner", "full_d2h", "rank", "cost(preds)", "time_s"]]
  for m in ("full", "sha", "race"):
    win, cost, t = res[m]
    out.append([m, win, "%.3f" % fullds[win], rank[win], cost, "%.3f" % t])
  print(os.path.basename(file),
        " pos=%s cands=%d test=%d  (rank 0 = true best)" %
        (c.pos, len(cands), len(test)))
  print2d(out)

if __name__ == "__main__":
  a   = sys.argv
  get = lambda f, d: of(a[a.index(f)+1]) if f in a else d
  fft.the.file = file = get("-f", os.path.expandvars("$DOOT/fairnez/adult.csv"))
  random.seed(fft.the.seed)
  if   "--csvhead" in a: print(",".join(CSVHEAD))
  elif "--finalhead" in a: print(",".join(FINALHEAD))
  elif "--final" in a: finalRun(file, get("-t", 100), get("-P", "auto"),
                                get("-r", 10), get("-S", 50), asCsv=True)
  elif "--csv" in a: compare(file, get("-t", 100), get("-P", "auto"),
                             get("-r", 10), get("-S", 50), asCsv=True)
  elif "--sha" in a: compare(file, get("-t", 100), get("-P", "auto"),
                             get("-r", 10), get("-S", 50))
  else:              main(file, get("-t", 100), get("-P", "auto"), get("-r", 10))
