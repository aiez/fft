#!/usr/bin/env python3 -B
"""r.py: fast-frugal multi-objective tree.
(c) 2025 Tim Menzies timm@ieee.org, MIT license

Options:
 -s seed     seed=1234567891
 -p dist exp  p=2
 -b bins      bins=7
 -d depth     depth=4
 -R Round     Round=2
 -f file      file=../optimiz/auto93.csv
"""
import sys, re, random
from math import sqrt, exp
from types import SimpleNamespace as o
BIG = 1E32

#-- 1. Columns ---------------------------------------------
def symp(x): return isinstance(x,dict)
def Sym()  : return {}
def Num()  : return (0,0,0)       # (n, mu, m2)

def sd(num):
  n, m2 = num[0], num[2]
  return 0 if n < 2 else sqrt(max(0, m2)/(n-1))

def norm(num,v):
  z = (v - num[1])/ (sd(num) + 1/BIG)
  return 1/(1+ exp(-1.7*max(-3, min(3, z))))

#-- 2. Data ------------------------------------------------
def Data(src=[]):
  def roles(names):
    i.names = names
    for at, s in enumerate(names):
      z = s[-1]
      i.cols[at] = Sym() if s[0].islower() else Num()
      if z in "-+!":
        i.y += [at]
        i.goal[at] = z=="+"
      elif z != "X": i.x += [at]
      if z=="!": i.klass=at

  src = iter(src)
  i=o(names=[],klass=None,x=[],y=[],goal={},cols={},rows=[])
  roles(next(src))
  for row in src: adds(i,row)
  return i

def adds(data, row, w=1):
  (data.rows.append if w==1 else data.rows.remove)(row)
  for at, v in enumerate(row):
    data.cols[at] = add(data.cols[at], v, w)

def add(c, v, w=1):
  if v == "?": return c
  if symp(c): c[v] = c.get(v, 0) + w; return c
  n, mu, m2 = c
  n += w
  if n <= 0: return Num()
  d   = v - mu
  mu += w * d / n
  m2 += w * d * (v - mu)
  return (n, mu, m2)

#-- 3. Discetization ---------------------------------------
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
    score = l[2] + merge(tot, l, -1)[2]
    yield (score, at, hi[k], hi[k], l)

def numCuts(bins, tot, hi, at):
  l = Num()
  for k in sorted(bins)[:-1]:
    l = merge(l, bins[k])
    score = l[2] + merge(tot, l, -1)[2]
    yield (score, at, -BIG, hi[k], l)

def merge(n1, n2, w=1):
  if symp(n1):
     return {k: n1.get(k,0) + w*n2.get(k,0) for k in n1 | n2}
  (an,amu,am2), (bn,bmu,bm2) = n1, n2
  n = an + w*bn
  if n <= 0: return Num()
  mu = (an*amu + w*bn*bmu) / n
  d  = bmu - amu
  m2 = am2 + w*bm2 + w*d*d*an*bn / n
  return (n, mu, m2)

#-- 4. build a tree ----------------------------------------
def disty(data, row):
  p, s, n = the.p, 0, 0
  for at in data.y:
    if (v := row[at]) == "?": continue
    s += abs(norm(data.cols[at], v) - data.goal[at])**p
    n += 1
  return (s/n)**(1/p) if n else 0

def statOf(rows, y):
  s = Num()
  for r in rows: s = add(s, y(r))
  return s

def has(v, lo, hi): return v == "?" or lo <= v <= hi 

def rest(rows, at, lo, hi):    
  return [r for r in rows if not has(r[at], lo, hi)]

def splits(data, y, root):
  floor = len(root.rows)**.33
  cs = [c for c in cuts(data, data.rows, y) if c[4][0] > floor]
  if cs:
    for bit, pick in enumerate((min, max)):
      _, at, lo, hi, leaf = pick(cs, key=lambda c: c[4][1])
      no = rest(data.rows, at, lo, hi)
      if no:
        yield bit, o(at=at, lo=lo, hi=hi, left=leaf), no
  
def grows(data, y, root, d=0):
  any = False
  if d < the.depth:
    for bit, nd, no in splits(data, y, root):
      for bias,right in grows(Data([data.names]+no),y,root,d+1):
        any = True
        yield str(bit)+bias, o(at=nd.at, lo=nd.lo, hi=nd.hi,
                               left=nd.left, right=right)
  if not any:
    yield "", statOf(data.rows, y)

#-- 5. use a tree ------------------------------------------
def predict(t, row):
  while not isinstance(t, tuple):
    t = t.left if has(row[t.at], t.lo, t.hi) else t.right
  return t[1]

def tune(cands, rows, y):
  err = lambda t: sum(abs(y(r)-predict(t,r))
                      for r in rows)/len(rows)
  return min(cands, key=err)

def show(data, t):
  if isinstance(t, tuple):
    print("%-33s leaf  d2h %.2f n=%d" % ("", t[1], t[0]))
    return
  nm = data.names[t.at]
  if   t.lo == t.hi: c = f"{nm} == {t.lo}"
  elif t.lo == -BIG: c = f"{nm} <= {qty(t.hi)}"
  else:              c = f"{nm} >= {qty(t.lo)}"
  L = t.left
  print("if %-30s then d2h %.2f n=%d" % (c, L[1], L[0]))
  show(data, t.right)

#-- 6. io --------------------------------------------------
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

#-- 7. Dests/ demos ---------------------------------------
def test_grows(repeats=10, k=100):
  import time
  rows = list(csv(the.file))
  names, body = rows[0], rows[1:]
  t = time.perf_counter()
  for _ in range(repeats):
    data = Data([names] + random.sample(body, k))
    y    = lambda r: disty(data, r)
    trees = list(grows(data, y, data))
  t = time.perf_counter() - t
  print("%dx (sample %d, %d trees): %.3f s  -> %.1f ms/round"
        % (repeats, k, len(trees), t, t/repeats*1000))

def test_trees():
  data = Data(csv(the.file))
  y    = lambda r: disty(data, r)
  err  = lambda t: sum(abs(y(r)-predict(t,r)) 
                       for r in data.rows)/len(data.rows)
  for i, (bias, t) in enumerate(grows(data, y, data), 1):
    print("===== tree %2d   bias %-5s   err %.3f =====" % (
         i, bias, err(t)))
    show(data, t)
    print()

#-- 8. Start-up --------------------------------------------
the=o(**{k:of(v) for k,v in re.findall(r"(\w+)=(\S+)", __doc__)})
if __name__ == "__main__":
  random.seed(the.seed)
  if   "--grows" in sys.argv: test_grows()
  elif "--trees" in sys.argv: test_trees()
  else:
    data  = Data(csv(the.file))
    y     = lambda r: disty(data, r)
    cands = [t for _, t in grows(data, y, data)]
    show(data, tune(cands, data.rows, y))
