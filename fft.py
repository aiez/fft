#!/usr/bin/env python3 -B
"""fft.py: fast-frugal multi-objective tree.
(c) 2025, Tim Menzies timm@ieee.org, MIT license

Options:
 -s seed     seed=1234567891
 -p dist     distance exponent  p=2
 -b bins     bins=7
 -d depth    depth=4
 -R Round    Round=2
 -f file     file=$DOOT/optimiz/auto93.csv

"""
import sys, re, random, os
from math import sqrt, exp
from types import SimpleNamespace as o
BIG = 1E32

#-- 1. Columns --------------------------------------------------
# if Num as tuples +  memos on disty, then (66..100)% faster s
# for small to large files.  
def Sym()  : return {}
def Num(n=0, mu=0, m2=0): return (n, mu, m2) 

def n_(x) : return x[0]       
def mu_(x): return x[1]      
def m2_(x): return x[2]     

def symp(x): return isinstance(x,dict)
def nump(x): return isinstance(x,tuple)

def sd(num):
  n, mu, m2 = num
  return 0 if n < 2 else sqrt(max(0, m2)/(n-1))

def welford(num, v, w=1):    
  n, mu, m2 = num
  if (n := n + w) <= 0: return Num()
  d = v - mu; mu += w * d / n
  return n, mu, m2 + w * d * (v - mu)

def norm(num,v):
  n, mu, m2 = num
  z = (v - mu)/ (sd(num) + 1/BIG)
  return 1/(1+ exp(-1.7*max(-3, min(3, z))))

def merge(i, j, w=1):
  if symp(i):
     return {k: i.get(k,0) + w*j.get(k,0) for k in i | j}
  i_n,i_mu,i_m2 = i
  j_n,j_mu,j_m2 = j
  n = i_n + w*j_n
  if n <= 0: return Num()
  mu = (i_n*i_mu + w*j_n*j_mu) / n
  d  = j_mu - i_mu
  m2 = i_m2 + w*j_m2 + w*d*d*i_n*j_n / n
  return Num(n, mu, m2)

#-- 2. Data -----------------------------------------------------
def Data(src=[]):
  def roles(names):
    it.names = names
    for at, s in enumerate(names):
      z = s[-1]
      it.cols[at] = Sym() if s[0].islower() else Num()
      if z in "-+!":
        it.y += [at]
        it.goal[at] = z=="+"
      elif z != "X": it.x += [at]
      if z=="!": it.klass=at

  src = iter(src)
  it=o(names=[],klass=None, x=[], y=[], goal={}, cols={},rows=[])
  roles(next(src))
  return adds(src, it)

def add(it, v, w=1):
  if v != "?":
    if   symp(it): it[v] = it.get(v, 0) + w
    elif nump(it): it = welford(it, v, w)
    else:
      (it.rows.append if w==1 else it.rows.remove)(v)
      for at, x in enumerate(v):
        it.cols[at] = add(it.cols[at], x, w)
  return it

def adds(src, it=None):
  if it is None: it = Num()
  for x in src: it = add(it, x)
  return it

##-- 3. Discetization --------------------------------------------
def cuts(data, rows, y):
  ys = [y(r) for r in rows]
  for at in data.x:
    c, tot, bins, hi = data.cols[at], Num(), {}, {}
    for r, y1 in zip(rows, ys):
      if (v := r[at]) == "?": continue
      k  = v if symp(c) else int(the.bins*norm(c,v))
      bins[k] = add(bins.get(k) or Num(), y1)
      tot     = add(tot, y1)
      hi[k]   = v if symp(c) else max(hi.get(k, -BIG), v)
    yield from (symCuts if symp(c) else numCuts)(bins,tot,hi,at)

def symCuts(bins, tot, hi, at):
  for k, l in bins.items():
    score = m2_(l) + m2_(merge(tot, l, -1))
    yield (score, at, hi[k], hi[k], l)

def numCuts(bins, tot, hi, at):
  l = Num()
  for k in sorted(bins)[:-1]:
    l = merge(l, bins[k])
    score = m2_(l) + m2_(merge(tot, l, -1))
    yield (score, at, -BIG, hi[k], l)

#-- 4. build a tree ---------------------------------------------
def disty(data, row):
  p, s, n = the.p, 0, 0
  for at in data.y:
    if (v := row[at]) == "?": continue
    s += abs(norm(data.cols[at], v) - data.goal[at])**p
    n += 1
  return (s/n)**(1/p) if n else 0


def has(v, lo, hi): return v == "?" or lo <= v <= hi

def trees(data, y=None):
  # ifan (FFTrees, Phillips et al'17): rank cues ONCE on root, keep
  # the top `depth`, then fan over 2^depth exit-direction patterns.
  # y + cue ranking root-scoped; a bin means the same root-scale notch.
  y     = memo(y or (lambda r: disty(data, r)))
  best  = {}                          # lowest-score (best) cut per cue
  for c in cuts(data, data.rows, y):
    if c[1] not in best or c[0] < best[c[1]][0]: best[c[1]] = c
  order = [c[1:4] for c in sorted(best.values())[:the.depth]]   # (at,lo,hi)

  def grow(rows, lvl):                # at each cue, fan BOTH exit directions
    any = False
    if lvl < len(order) and rows:
      at, lo, hi = order[lvl]
      for yes in (False, True):
        out  = [r for r in rows if has(r[at], lo, hi) == yes]   # exits here
        cont = [r for r in rows if has(r[at], lo, hi) != yes]   # falls through
        if out and cont:
          for bias, right in grow(cont, lvl + 1):
            any = True
            yield str(int(yes)) + bias, o(at=at, lo=lo, hi=hi, yes=yes,
                     left=adds(y(r) for r in out), right=right)
    if not any:
      yield "", adds(y(r) for r in rows)

  return grow(data.rows, 0)

#-- 5. use a tree -----------------------------------------------
def predict(t, row):
  while not nump(t):              # left = exit leaf, right = fall-through
    t = t.left if has(row[t.at], t.lo, t.hi) == t.yes else t.right
  return mu_(t)

def tune(cands, rows, y):
  err = lambda t: sum(abs(y(r)-predict(t,r))
                      for r in rows)/len(rows)
  return min(cands, key=err)

def show(data, t):
  if nump(t):
    print("%-33s leaf  d2h %.2f n=%d" % ("", mu_(t), n_(t)))
    return
  nm = data.names[t.at]                       # negate cue if exit is rest-side
  if   t.lo == t.hi: c = f"{nm} {'==' if t.yes else '!='} {t.lo}"
  elif t.lo == -BIG: c = f"{nm} {'<=' if t.yes else '>'} {qty(t.hi)}"
  else:              c = f"{nm} {'>=' if t.yes else '<'} {qty(t.lo)}"
  L = t.left
  print("if %-30s then d2h %.2f n=%d" % (c, mu_(L), n_(L)))
  show(data, t.right)

#-- 6. lin -------------------------------------------------------
def memo(fn):               # cache fn(row) by row identity
  cache = {}
  def f1(row):
    if (k := id(row)) not in cache: cache[k] = fn(row)
    return cache[k]
  return f1

def of(z):
  for f in (int, float):
    try: return f(z)
    except: pass
  z = z.strip()
  return {'True':True,'False':False,'None':None}.get(z, z)

def csv(file):
  for ln in open(file):
    ln = ln.strip()
    if ln and ln[0] != "#":
      yield [of(x) for x in ln.split(",")]

def qty(v):
  if isinstance(v, float):
    return int(v) if int(v)==v else round(v, the.Round)
  return v

#-- 7. Dests/ demos --------------------------------------------
def test_main():
  data  = Data(csv(the.file))
  y     = memo(lambda r: disty(data, r))
  cands = [t for _, t in trees(data, y)]
  show(data, tune(cands, data.rows, y))

def test_grows(repeats=10, k=100):
  import time
  rows = list(csv(the.file))
  names, body = rows[0], rows[1:]
  t = time.perf_counter()
  for _ in range(repeats):
    data = Data([names] + random.sample(body, k))
    cands = list(trees(data))
  t = time.perf_counter() - t
  print("%dx (sample %d, %d trees): %.3f s  -> %.1f ms/round"
        % (repeats, k, len(cands), t, t/repeats*1000))

def test_trees():
  data = Data(csv(the.file))
  y    = memo(lambda r: disty(data, r))
  err  = lambda t: sum(abs(y(r)-predict(t,r)) 
                       for r in data.rows)/len(data.rows)
  for i, (bias, t) in enumerate(trees(data), 1):
    print("===== tree %2d   bias %-5s   err %.3f =====" % (
         i, bias, err(t)))
    show(data, t)
    print()

#-- 8. Start-up -------------------------------------------------
os.environ.setdefault("DOOT",
  os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
the=o(**{k:of(v) for k,v in re.findall(r"(\w+)=(\S+)", __doc__)})
the.file=os.path.expandvars(the.file)
random.seed(the.seed)

if __name__ == "__main__":
  for f, v in zip(sys.argv, sys.argv[1:]):
    for k in vars(the):
      if f == "-"+k[0]: setattr(the, k, of(v))
  if   "--grows" in sys.argv: test_grows()
  elif "--trees" in sys.argv: test_trees()
  else: test_main()
