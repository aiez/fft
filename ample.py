#!/usr/bin/env python3 -B
"""
fft1.py, random-cluster forest engine + confusion (library)
(c) 2025, Tim Menzies <timm@ieee.org>, MIT license

Generic services: Data, disty, tree (random axis-cut cluster),
treeLeaf (router), treeShow (printer), confused (pd/pf/.. from
(want,got) pairs), cli/the. No task knowledge -- see compas.py.

Options:
 -s --seed  random seed       seed=1234567891
 -p --p     distance exponent p=2
 -R --Round repr decimals     Round=2
 -S --stop  leaf size         stop=None
 -n --N     demo row sample   N=50
 -f --file  data file
            file=/Users/timm/gits/moot/optimize/misc/auto93.csv

eg: python3 fft1.py -f FILE -n 50 --tree
"""
import random, math, sys, re
from functools import reduce
BIG = math.inf

# ## constructors -----------------------------------------------
def Num(at=0, txt=""):
  return o(it=Num, at=at, txt=txt, n=0, mu=0, m2=0, sd=0,
           lo=BIG, hi=-BIG, goal=0 if txt.endswith("-") else 1)

def Sym(at=0, txt=""):
  return o(it=Sym, at=at, txt=txt, n=0, has={})

def Data(src=[]):
  return adds(src, o(it=Data, cols=None, rows=[]))

def Cols(names):
  cols,x,y,klass = [],[],[],None
  for at, s in enumerate(names):
    z = s[-1]
    cols += [col := (Num if s[0].isupper() else Sym)(at, s)]
    if z in "-+!":
      y += [col]      
      if z == "!": klass = col
    elif z != "X":  x += [col]   
  return o(
    it=Cols, all=cols, x=x,y=y, names=names, klass=klass or y[0])

def Tree(): pass                    # interior tag (leaf=Data)

# ## add (one polymorphic) --------------------------------------
def add(i, v):
  if i.it is Data:
    if not i.cols: i.cols = Cols(v)
    else:
      i.rows += [v]
      for c in i.cols.all: add(c, v[c.at])
  elif v == "?": pass
  elif i.it is Sym:
    i.n += 1; i.has[v] = i.has.get(v, 0) + 1
  else:                                # add a number
    i.n += 1; d = v - i.mu
    i.mu += d / i.n
    i.m2 += d * (v - i.mu)
    i.sd = (i.m2/(i.n-1))**0.5 if i.n > 1 else 0
    if v < i.lo: i.lo = v
    if v > i.hi: i.hi = v
  return v

def adds(src, it=None):
  for x in iter(src): add((it := it or Num()), x)
  return it

def clone(root, rows):
  return adds(rows, adds([root.cols.names],Data()))

# ## merge (s=+1 add, s=-1 subtract) ----------------------------
def merge(a, b, s=1):
  if a.it is Sym:
    has = dict(a.has)
    for k, v in b.has.items():
      has[k] = has.get(k, 0) + s*v
    has = {k: v for k, v in has.items() if v > 0}
    return o(it=Sym, at=a.at, txt=a.txt, n=a.n + s*b.n, has=has)
  n = a.n + s*b.n
  if n <= 0: return Num(a.at, a.txt)
  mu = (a.n*a.mu + s*b.n*b.mu) / n
  d  = b.mu - a.mu
  m2 = max(0, a.m2 + s*b.m2 + s*d*d*a.n*b.n / n)
  c  = Num(a.at, a.txt)
  c.n, c.mu, c.m2 = n, mu, m2
  c.lo, c.hi = min(a.lo, b.lo), max(a.hi, b.hi)
  return c

# ## metrics ----------------------------------------------------
def norm(c, v):
  return v if v=="?" else (v - c.lo) / (c.hi - c.lo + 1E-32)

def binOf(c, x, bins):               # key = cut value (rep edge for Num)
  if c.it is Sym: return x
  k = min(bins-1, int(bins*(x-c.lo)/(c.hi-c.lo+1E-32)))
  return c.lo + (k+1)*(c.hi-c.lo)/bins

def disty(d, r):
  s, n, p = 0, 0, the.p
  for c in d.cols.y:
    n += 1; s += abs(norm(c, r[c.at]) - c.goal)**p
  return (s/n)**(1/p) if n else 0

def distx(d, r1, r2):                # over x-cols, missing-tolerant
  s, n, p = 0, 0, the.p
  for c in d.cols.x:
    n += 1; v1, v2 = r1[c.at], r2[c.at]
    if v1=="?" and v2=="?": s += 1; continue
    if c.it is Sym: s += 0 if v1==v2 else 1
    else:
      v1 = norm(c,v1) if v1!="?" else (0 if norm(c,v2)>.5 else 1)
      v2 = norm(c,v2) if v2!="?" else (0 if v1>.5 else 1)
      s += abs(v1-v2)**p
  return (s/n)**(1/p)

# ## fmtree (fastmap-split, stochastic poles, train-worth) -----
def FMTree(): pass                   # tag for fmtree inner nodes

def fmtree(root, stop=None, k=30):
  stop = stop or int(len(root.rows)**.5)
  def far(ref, some):
    return max(some, key=lambda r: distx(root, ref, r))
  def grow(rows):
    if len(rows) <= stop:
      mu = sum(disty(root, r) for r in rows)/max(1, len(rows))
      return o(it=Data, rows=rows, worth=mu)
    some = random.sample(rows, min(k, len(rows)))
    A = far(random.choice(some), some); B = far(A, some)
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
# edges, Sym -> category) into a Num of yfun(row) per bin, then
# sweeps for the min child-variance cut: total = merge(all bins),
# each split's R = merge(total, L, -1) (unmerge); impurity =
# L.m2+R.m2. default yfun = klass; caller can swap target.
def cutgo(c, x, v):                  # x -> left branch?
  return x != "?" and (x == v if c.it is Sym else x <= v)

def tree(root, stop=None, yfun=None, bins=7):
  yfun = yfun or (lambda r: r[root.cols.klass.at])
  stop = stop or the.stop or len(root.rows)**.5

  def cuts(c, rows):                 # yield (impurity, cut-value) over bins
    hist = {}
    for r in rows:
      if (x := r[c.at]) != "?":
        add(hist.setdefault(binOf(c, x, bins), Num()), yfun(r))
    bs = sorted(hist.items())
    if not bs: return
    total = reduce(merge, (v for _, v in bs))
    if c.it is Sym:
      for k, v in bs:
        R = merge(total, v, -1)
        if v.n and R.n: yield v.m2 + R.m2, k
    else:
      L = Num()
      for k, v in bs[:-1]:
        L = merge(L, v)
        R = merge(total, L, -1)
        if L.n and R.n: yield L.m2 + R.m2, k

  def grow(rows):
    if len(rows) <= stop: return clone(root, rows)
    for _ in range(10):              # retry till a real 2-way cut
      c = random.choice(root.cols.x)
      _, v = min(cuts(c, rows), default=(BIG, "?"))
      if v == "?": continue
      ok, no = [], []
      for r in rows:
        (ok if cutgo(c, r[c.at], v) else no).append(r)
      if ok and no: break
    else:
      return clone(root, rows)       # no good cut -> leaf
    return o(it=Tree, at=c.at, cut=v, left=grow(ok), right=grow(no))

  return grow(root.rows)

def treeLeaf(root, t, row):          # route row to leaf
  while t.it is Tree:
    c = root.cols.all[t.at]
    t = t.left if cutgo(c, row[t.at], t.cut) else t.right
  return t

def leaves(t):                       # yield the leaf Datas of a tree
  if t.it is Data: yield t
  else: yield from leaves(t.left); yield from leaves(t.right)

def treeShow(root, t):               # ygoal, goal means, tree
  ys  = [c for c in root.cols.y if c.it is Num]
  rowsOf = lambda t: t.rows if t.it is Data else \
                     rowsOf(t.left) + rowsOf(t.right)
  dy  = lambda rs: adds(disty(root,r) for r in rs).mu
  hdr = "".join("%7s" % c.txt for c in ys)
  print("%6s%s %5s  tree" % ("ygoal", hdr, "n"))
  def show(t, lvl, txt):
    rs = rowsOf(t)
    g = "".join("%7.1f" %
                adds(r[c.at] for r in rs if r[c.at]!="?").mu
                for c in ys)
    print("%6.2f%s %5d  %s%s" %
          (dy(rs), g, len(rs), "|  "*lvl, txt))
    if t.it is Tree:
      c  = root.cols.all[t.at]
      lo, hi = ("<="," >") if c.it is Num else ("==","!=")
      for kid, op in sorted([(t.left, lo), (t.right, hi)],
                       key=lambda k: dy(rowsOf(k[0]))):
        show(kid, lvl+1, "%s %s %s" % (c.txt, op, t.cut))
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
      random.seed(the.seed); fn()
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

def test_tree():                # tree on N rows of -f
  head, *body = csv(the.file)
  rows = random.sample(body, min(the.N, len(body)))
  d = Data([head] + rows)
  treeShow(d, tree(d))

def test_fmtree():              # 30 fmtrees, runtime + worth dist
  import time
  head, *body = csv(the.file)
  rows = random.sample(body, min(the.N, len(body)))
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
random.seed(the.seed)
if __name__ == "__main__":
  cli(the, __doc__, globals())
