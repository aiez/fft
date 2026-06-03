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
 -b --bins  numeric bins      bins=7
 -F --Few   a few samples     Few=64
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
import sys, re
from math import inf, log, log2, sqrt
from random import random, randrange, choice, sample, seed
BIG = inf
# ## constructors -----------------------------------------------
def symp(col) : return type(col) is dict
def nump(col) : return type(col) is list

def Data(src):
  src = iter(src)
  cols, w, x, y, klass = {}, {}, {}, {}, None 
  for at, s in enumerate(names = next(src)):
    z = s[-1]
    cols[at] = col = [] if s[0].isupper() else {}  
    if   z in "-+!" : y[at]=col; w[at]=z=="+"
    elif z != "X"   : x[at]=col
    if z=="!":klass=at
  meta = o(cols=cols,names=names, w=w, x=x, y=y, klass=klass)
  return ok(adds(src, o(rows=[], stale=False, cols=meta)))

def clone(root, rows=[]): return Data([root.cols.names] + rows)

def adds(src, data):
  for row in src: add(data, row)
  return data

def add(data, row):
  data.stale = True                  
  data.rows += [row]
  for at, col in data.cols.all.items():
    if (x := row[at]) != "?":
      if symp(col): col[x] = col.get(x, 0) + 1  
      elif len(col) < the.Some: col += [x]  
      elif random() < the.Some / len(data.rows):  
        col[randrange(the.Some)] = x
  return row

def ok(data):                  
  if data.stale: update(data); data.stale=False
  return data

def udpate(data):
  for col in data.cols.all.values():
    if nump(col): col.sort()

# ## methods ----------------------------------------------------
def mu(lst): return sum(lst)/len(lst) if a else 0

def mid(col): 
  return col[len(col)//2] if nump(col) else max(col,key=col.get)

def spread(col):
  if symp(col):
    N = sum(col.values()) or 1
    return -sum(v/N*log2(v/N) for v in col.values() if v)
  n = len(col)
  if n < 2:  return 0
  if n >= 20: return (col[int(.9*n)] - col[int(.1*n)]) / 2.56
  return (col[-1] - col[0]) / (0.34 + 1.13*log(n))   # tiny n: range

# ## metrics ----------------------------------------------------
def norm(lst, v):            
  a, b = lst[0], lst[-1]
  return v if v == "?" else (v - a) / (b - a + 1/BIG)

def disty(data, row):
  s, n, p = 0, 0, the.p
  for at,lst in ok(data).cols.y.items():
    n += 1
    s += abs(norm(lst, row[at]) - data.cols.w[at])**p
  return (s/n)**(1/p) if n else 0

def distx(data, row1, row2):         # over x-cols, missing-tolerant
  s, n, p = 0, 0, the.p
  for at, col in ok(data).cols.x.items():
    n += 1
    v1, v2 = row1[at], row2[at]
    if v1=="?" and v2=="?": s += 1; continue
    if symp(col): s += 0 if v1==v2 else 1
    else:
      v1, v2 = norm(col,v1), norm(col,v2)
      v1 = v1 if v1!="?" else (0 if v2>.5 else 1)
      v2 = v2 if v2!="?" else (0 if v1>.5 else 1)
      s += abs(v1-v2)**p
  return (s/n)**(1/p)

def neighbors(data,rows,row):
  return sorted(rows, key=lambda r:distx(data,row,r))

def far(data, rows, row):          
  return neighbors(Data,rows,rows)[int(.9*len(rows)]

def dim(data, rows):           
  some = sample(rows, min(len(rows), the.Few))
  a    = far(data, some, choice(some))
  b    = far(data, some, a)
  gap  = lambda r1, r2: distx(data, r1, r2)
  c    = gap(a, b) + 1/BIG       
  out  = []
  for row in rows:
    da   = gap(row, a)
    x    = (da*da + c*c - gap(row, b)**2) / (2*c)
    y    = sqrt(max(0, da*da - x*x))          
    out += [(x, y, row)]
  return a,b,c,sorted(out)

def grid(data, rows):
  a, b, c, out = dim(data, rows)        
  N = len(out)
  G = max(1, int(N**0.25))      # G*G ~ sqrt(N) cells
  strips = chunks(out, N // G)  # equal-count x-strips
  def cellsOf(strip):
    cs = chunks(sorted(strip, key=lambda t:t[1]), len(strip)//G)
    return ([cell[0][1] for cell in cs[1:]],     # y lower-edges
            [clone(data, [t[2] for t in cell]) for cell in cs])
  xedges = [s[0][0] for s in strips[1:]]         # x lower-edges
  cells  = [cellsOf(s) for s in strips]
  return o(a=a, b=b, c=c, xedges=xedges, cells=cells)

def gridFind(data, g, row):
  d  = lambda r1, r2: distx(data, r1, r2)
  da = d(row, g.a)
  x  = (da*da + g.c*g.c - d(row, g.b)**2) / (2*g.c)
  y  = sqrt(max(0, da*da - x*x))
  yedges, col = g.cells[bisect(g.xedges, x)]   # which x-strip
  return col[bisect(yedges, y)]                # which y-cell -> Data

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
def chunks(xs, size):                   
  return [xs[i:i+size] for i in range(0, len(xs), max(1, size))]

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
      yield [of(x.strip()) for x in ln.split(",")]

def of(z):
  for f in (int, float):
    try: return f(z)
    except: pass
  z = z.strip()
  return {'True':True,'False':False,'None':None}.get(z,z)

def settings(doc):              # key=val defaults from doc
  return o(**{k:of(v) for k,v in re.findall(r'(\w+)=(\S+)',doc)})

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
      elif j+1 < len(sys.argv): v = of(sys.argv[j+1])
      the[k] = v
    elif re.match(r"-\D", s):
      sys.exit("bad flag: %s\n%s" % (s, doc))
  return the

# ## egs (run via --name) --------------------------------------
def test_the():                 # show the config
  print(the)

def test_stats():               # mid/spread per col of -f
  head, *body = csv(the.file)
  d = Data([head] + body)
  print("%-14s %5s %10s %10s" % ("col","n","mid","spread"))
  for at, c in d.cols.all.items():
    n = len(c) if nump(c) else sum(c.values())
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
