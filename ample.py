#!/usr/bin/env python3 -B
"""
fft1.py, random-cluster forest engine + confusion (library)
(c) 2025, Tim Menzies <timm@ieee.org>, MIT license

Generic services: Data, disty, tree (random axis-cut cluster),
treeLeaf (router), treeShow (printer), confused (pd/pf/.. from
(want,got) pairs), cli/the. No task knowledge -- see compas.py.

Columns are bare containers: a Num is a list (reservoir of its
values), a Sym is a dict (value->count). No Num/Sym structs --
dispatch on type(c). Per-column meta (name, goal, x/y role) lives
in cols.names / cols.x / cols.y, keyed by position `at`.

Options:
 -s --seed  random seed       seed=1234567891
 -p --p     distance exponent p=2
 -R --Round repr decimals     Round=2
 -S --stop  leaf size         stop=None
 -n --N     demo row sample   N=50
 -c --Some  reservoir cap     Some=256
 -f --file  data file
            file=/Users/timm/gits/moot/optimize/misc/auto93.csv

eg: python3 fft1.py -f FILE -n 50 --tree
"""
import math, sys, re
from random import random, randrange, choice, sample, seed
BIG = math.inf

# ## constructors -----------------------------------------------
# a column is just its container: list = Num, dict = Sym.
def is_num(c): return type(c) is list
def is_sym(c): return type(c) is dict

def Data(src=[]):
  return adds(src, o(it=Data, cols=None, rows=[], stale=False))

def Cols(names):                     # data lives in `all`; x/y ref it
  all, x, y, klass = [], {}, {}, None     # x,y = {at: container}
  for at, s in enumerate(names):
    all += [[] if s[0].isupper() else {}]   # Num=list, Sym=dict
    z = s[-1]
    if z in "-+!":
      y[at] = all[-1]                      # same mutable ref as all[at]
      if z == "!": klass = at
    elif z != "X": x[at] = all[-1]
  return o(it=Cols, all=all, names=names, x=x, y=y,
           klass=klass if klass is not None else next(iter(y), 0))

def Tree(): pass                     # interior tag (leaf=Data)

# ## add --------------------------------------------------------
def add(i, v):                       # i is a Data
  if not i.cols: i.cols = Cols(v); return v
  i.rows += [v]
  i.stale = True    # cached mid/spread now dirty
  for at, c in enumerate(i.cols.all):
    if (x := v[at]) != "?":
      if is_sym(c): c[x] = c.get(x, 0) + 1           # count
      elif len(c) < the.Some: c.append(x)            # reservoir fill
      elif random() < the.Some / len(i.rows):        # Vitter replace
        c[randrange(the.Some)] = x
  return v

def adds(src, it):
  for x in src: add(it, x)
  return it

def clone(root, rows):
  return adds(rows, adds([root.cols.names], Data()))

def ok(d):                           # sort Num reservoirs, clear stale
  if d.stale:
    for c in d.cols.all:
      if is_num(c): c.sort()
    d.stale = False
  return d

def mu(a): return sum(a)/len(a) if a else 0

def mid(c):                          # median (Num) | mode (Sym)
  return c[len(c)//2] if is_num(c) else max(c, key=c.get)

def spread(c):                       # ~sigma (Num) | entropy bits (Sym)
  if is_sym(c):
    N = sum(c.values()) or 1
    return -sum(v/N*math.log2(v/N) for v in c.values() if v)
  n = len(c)
  if n < 2:  return 0
  if n >= 20: return (c[int(.9*n)] - c[int(.1*n)]) / 2.56
  return (c[-1] - c[0]) / (0.34 + 1.13*math.log(n))   # tiny n: range

# ## metrics ----------------------------------------------------
def norm(c, v):                      # 0..1 via reservoir lo/hi (sorted)
  if v == "?": return v
  if not c:    return .5
  a, b = c[0], c[-1]
  return (v - a) / (b - a + 1E-32)

def binOf(c, x, bins):               # key = cut value (rep edge for Num)
  if is_sym(c) or not c: return x
  a, b = c[0], c[-1]
  k = min(bins-1, int(bins*(x-a)/(b-a+1E-32)))
  return a + (k+1)*(b-a)/bins

def goalOf(d, at):                   # 0 = minimize, 1 = maximize
  return 0 if d.cols.names[at].endswith("-") else 1

def disty(d, r):
  ok(d); s, n, p = 0, 0, the.p
  for at, c in d.cols.y.items():
    n += 1; s += abs(norm(c, r[at]) - goalOf(d, at))**p
  return (s/n)**(1/p) if n else 0

def distx(d, r1, r2):                # over x-cols, missing-tolerant
  ok(d); s, n, p = 0, 0, the.p
  for at, c in d.cols.x.items():
    n += 1; v1, v2 = r1[at], r2[at]
    if v1=="?" and v2=="?": s += 1; continue
    if is_sym(c): s += 0 if v1==v2 else 1
    else:
      v1 = norm(c,v1) if v1!="?" else (0 if norm(c,v2)>.5 else 1)
      v2 = norm(c,v2) if v2!="?" else (0 if v1>.5 else 1)
      s += abs(v1-v2)**p
  return (s/n)**(1/p)

# ## fmtree (fastmap-split, stochastic poles, train-worth) -----
def FMTree(): pass                   # tag for fmtree inner nodes

def fmtree(root, stop=None, k=30):
  ok(root)
  stop = stop or int(len(root.rows)**.5)
  def far(ref, some):
    return max(some, key=lambda r: distx(root, ref, r))
  def grow(rows):
    if len(rows) <= stop:
      return o(it=Data, rows=rows,
               worth=mu([disty(root, r) for r in rows]))
    some = sample(rows, min(k, len(rows)))
    A = far(choice(some), some); B = far(A, some)
    c = distx(root, A, B) + 1e-32
    proj = lambda r:(distx(root,r,A)**2 + c*c
                     - distx(root,r,B)**2)/(2*c)
    keyed = [(proj(r), r) for r in rows]
    keyed.sort(key=lambda x: x[0])
    m = len(keyed)//2
    L = grow([r for _,r in keyed[:m]])
    R = grow([r for _,r in keyed[m:]])
    return o(it=FMTree, A=A, B=B, c=c, cut=keyed[m][0],
             left=L, right=R, worth=max(L.worth, R.worth))
  return grow(root.rows)

def fmleaf(root, t, row):
  while t.it is FMTree:
    pr = (distx(root,row,t.A)**2 + t.c**2
          - distx(root,row,t.B)**2)/(2*t.c)
    t = t.left if pr <= t.cut else t.right
  return t

def fmleaves(t):                     # yield leaf Datas
  if t.it is Data: yield t
  else: yield from fmleaves(t.left); yield from fmleaves(t.right)

# ## tree (random attr, min-variance cut) ----------------------
# ONE random x-col/node. cuts() bins rows (binOf: Num -> `bins`
# edges, Sym -> category), collecting yfun(row) per bin, then
# sweeps for the min child-variance cut. Variance is recomputed
# straight from the per-side value lists (m2 = sum sq-dev) -- no
# struct, no subtractive merge. default yfun = klass value.
def cutgo(c, x, v):                  # x -> left branch?
  return x != "?" and (x == v if is_sym(c) else x <= v)

def m2(a):                           # sum of squared deviations
  if len(a) < 2: return 0
  m = sum(a)/len(a)
  return sum((x-m)**2 for x in a)

def tree(root, stop=None, yfun=None, bins=7):
  ok(root)
  yfun = yfun or (lambda r: r[root.cols.klass])
  stop = stop or the.stop or len(root.rows)**.5

  def cuts(c, rows, at):             # yield (impurity, cut-value)
    hist = {}
    for r in rows:
      if (x := r[at]) != "?":
        hist.setdefault(binOf(c, x, bins), []).append(yfun(r))
    bs = sorted(hist.items())
    if not bs: return
    if is_sym(c):
      for kk, v in bs:
        rest = [y for k2, vv in bs if k2 != kk for y in vv]
        if v and rest: yield m2(v) + m2(rest), kk
    else:
      flat = [y for _, v in bs for y in v]       # in cut order
      b = 0
      for kk, v in bs[:-1]:
        b += len(v); L, R = flat[:b], flat[b:]
        if L and R: yield m2(L) + m2(R), kk

  def grow(rows):
    if len(rows) <= stop: return clone(root, rows)
    for _ in range(10):              # retry till a real 2-way cut
      at = choice(list(root.cols.x))
      c  = root.cols.x[at]
      _, v = min(cuts(c, rows, at), default=(BIG, "?"))
      if v == "?": continue
      yes, no = [], []
      for r in rows:
        (yes if cutgo(c, r[at], v) else no).append(r)
      if yes and no: break
    else:
      return clone(root, rows)       # no good cut -> leaf
    return o(it=Tree, at=at, cut=v, left=grow(yes), right=grow(no))

  return grow(root.rows)

def treeLeaf(root, t, row):          # route row to leaf
  while t.it is Tree:
    t = t.left if cutgo(root.cols.all[t.at], row[t.at], t.cut) \
        else t.right
  return t

def leaves(t):                       # yield the leaf Datas of a tree
  if t.it is Data: yield t
  else: yield from leaves(t.left); yield from leaves(t.right)

def treeShow(root, t):               # ygoal, goal means, tree
  ok(root)
  ys = [at for at, c in root.cols.y.items() if is_num(c)]
  rowsOf = lambda t: t.rows if t.it is Data else \
                     rowsOf(t.left) + rowsOf(t.right)
  dy  = lambda rs: mu([disty(root, r) for r in rs])
  hdr = "".join("%7s" % root.cols.names[at] for at in ys)
  print("%6s%s %5s  tree" % ("ygoal", hdr, "n"))
  def show(t, lvl, txt):
    rs = rowsOf(t)
    g = "".join("%7.1f" % mu([r[at] for r in rs if r[at]!="?"])
                for at in ys)
    print("%6.2f%s %5d  %s%s" %
          (dy(rs), g, len(rs), "|  "*lvl, txt))
    if t.it is Tree:
      nm = root.cols.names[t.at]
      lo, hi = ("<="," >") if is_num(root.cols.all[t.at]) \
               else ("==","!=")
      for kid, op in sorted([(t.left, lo), (t.right, hi)],
                       key=lambda k: dy(rowsOf(k[0]))):
        show(kid, lvl+1, "%s %s %s" % (nm, op, t.cut))
  show(t, 0, "")

# ## confusion (pd, pf, ... from (want,got) pairs) --------------
def confused(pairs):            # pairs = [(want, got), ...]
  n, labels, N = {}, set(), len(pairs)
  for wg in pairs:
    n[wg] = n.get(wg, 0) + 1
    labels |= set(wg)
  out = {}
  for k in labels:                       # one-vs-rest per label
    tp = n.get((k, k), 0)
    fn = sum(c for (w,g),c in n.items() if w==k and g!=k)
    fp = sum(c for (w,g),c in n.items() if w!=k and g==k)
    tn = N - tp - fn - fp
    pd, pf = tp/(tp+fn+1e-32), fp/(fp+tn+1e-32)    # recall, fpr
    prec   = tp/(tp+fp+1e-32)
    out[k] = o(it=o, label=k, tp=tp, fp=fp, fn=fn, tn=tn,
               pd=pd, pf=pf, prec=prec, acc=(tp+tn)/(N+1e-32),
               f1=2*prec*pd/(prec+pd+1e-32))
  return out

# ## lib -------------------------------------------------------
class o(dict):
  __getattr__ = dict.get;__setattr__ = dict.__setitem__;
  def __repr__(i):
    def f(v):
      if isinstance(v,float):
        return int(v) if v==int(v) else round(v, the.Round)
      return v
    return "{" + " ".join(":%s %s" % (k, f(i[k])) for k in i
                          if str(k)[0]!="_") + "}"

def csv(file):
  for ln in open(file):
    ln = ln.strip()
    if ln and ln[0] != "#":
      yield [coerce(x.strip()) for x in ln.split(",")]

def coerce(z):
  for f in (int, float):
    try: return f(z)
    except: pass
  z = z.strip()
  return {'True':True,'False':False,'None':None}.get(z,z)

def settings(doc):              # key=val defaults from doc
  return o(**{k: coerce(v)
              for k, v in re.findall(r'(\w+)=(\S+)', doc)})

def cli(the, doc, egs={}):      # --x runs test_x()
  flags = {f: l.lstrip("-")
           for s,l in re.findall(r"(-\w) (--\w+)",doc)
           for f in (s,l)}
  for j, s in enumerate(sys.argv):
    if fn := egs.get("test_" + s.lstrip("-")):
      seed(the.seed); fn()
    elif k := flags.get(s):
      v = the[k]
      if isinstance(v, bool): v = not v
      elif j+1 < len(sys.argv): v = coerce(sys.argv[j+1])
      the[k] = v
    elif re.match(r"-\D", s):
      sys.exit("bad flag: %s\n%s" % (s, doc))
  return the

# ## egs (run via --name) --------------------------------------
def test_the():                 # show the config
  print(the)

def test_stats():               # mid/spread per col of -f
  head, *body = csv(the.file)
  d = ok(Data([head] + body))
  print("%-14s %5s %10s %10s" % ("col","n","mid","spread"))
  for at, c in enumerate(d.cols.all):
    n = len(c) if is_num(c) else sum(c.values())
    print("%-14s %5d %10s %10.3f" %
          (d.cols.names[at], n, str(mid(c)), spread(c)))

def test_tree():                # tree on N rows of -f
  head, *body = csv(the.file)
  rows = sample(body, min(the.N, len(body)))
  d = Data([head] + rows)
  treeShow(d, tree(d))

def test_fmtree():              # 30 fmtrees, runtime + worth dist
  import time
  head, *body = csv(the.file)
  rows = sample(body, min(the.N, len(body)))
  d    = Data([head] + rows)
  t0   = time.time()
  trees = [fmtree(d) for _ in range(30)]
  dt   = time.time() - t0
  ws   = sorted(t.worth for t in trees)
  best = min(trees, key=lambda t: t.worth)
  nLeaves = sum(1 for _ in fmleaves(best))
  print("# 30 fmtrees on %s (N=%d)" % (the.file.split("/")[-1], len(rows)))
  print("time         %.3fs  (%.1f ms/tree)" % (dt, 1000*dt/30))
  print("worth min    %.4f" % ws[0])
  print("worth med    %.4f" % ws[15])
  print("worth max    %.4f" % ws[-1])
  print("picked tree: worth=%.4f, %d leaves, stop=%d" %
        (best.worth, nLeaves, int(len(rows)**.5)))
  print("# all worths sorted:")
  for w in ws: print("  %.4f" % w)

# ## main -------------------------------------------------------
the = settings(__doc__)
seed(the.seed)
if __name__ == "__main__":
  cli(the, __doc__, globals())
