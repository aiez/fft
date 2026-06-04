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
import sys, re, bisect
from math import inf, log, log2
from random import random, randrange, sample, seed
from collections import defaultdict
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
      else: some(col, x, len(data.rows))
  return row

def some(lst, v, n):
  if len(lst) < the.Some: lst += [v]
  elif random() < the.Some/n: lst[randrange(the.Some)] = v
  return lst

def ok(data):                  
  if data.stale: update(data); data.stale=False
  return data

def udpate(data):
  for col in data.cols.all.values():
    if nump(col): col.sort()

# ## methods ----------------------------------------------------
def mid(col): 
  return col[len(col)//2] if nump(col) else max(col,key=col.get)

def spread(col):
  if symp(col):
    N = sum(col.values()) or 1
    return -sum(v/N*log2(v/N) for v in col.values() if v)
  n = len(col)
  if n < 2:  return 0
  if n >= 20: return (col[int(.9*n)] - col[int(.1*n)]) / 2.56
  return (col[-1] - col[0]) / (0.34 + 1.13*log(n))

# ## metrics ----------------------------------------------------
def norm(lst, v):            
  a, b = lst[0], lst[-1]
  return v if v == "?" else (v - a) / (b - a + 1/BIG)

def bin(col, v):                    # equal-freq bin id
  if symp(col): return v
  r = bisect.bisect_left(col, v)
  return min(the.bins-1, r*the.bins // max(1,len(col)))

def cutgo(col, v, cut):             # left-branch test
  if v == "?":  return True
  if symp(col): return v == cut
  return v <= cut

def disty(data, row):
  s, n, p = 0, 0, the.p
  for at,lst in ok(data).cols.y.items():
    n += 1
    s += abs(norm(lst, row[at]) - data.cols.w[at])**p
  return (s/n)**(1/p) if n else 0

def m2(a):                    
  if len(a) < 2: return 0
  m = sum(a)/len(a)
  return sum((x-m)**2 for x in a)

def cuts(at, col, rows, yfun):      # yield (impurity, cut)
  ys, hi = defaultdict(list), {}
  for r in rows:
    x = r[at]
    if x == "?": continue
    k = bin(col, x)
    ys[k] += [yfun(r)]
    if not symp(col): hi[k] = max(hi.get(k, x), x)
  fn = splitList if nump(col) else splitDict
  for k,yes,no in fn(sorted(ys),at,col):
    return m2(yes)+m2(no), k

def mergeSym(ks,at,col)
  for k in ks:
    a = ys[k]
    b = [y for j in ks if j != k for y in ys[j]]
    if a and b: yield k,a,b

def mergeNum(ks,at,col):
  flat = [y for k in ks for y in ys[k]]
  n = 0
  for k in ks[:-1]:
    n += len(ys[k])
    L, R = flat[:n], flat[n:]
    if L and R: yield L,R, hi[k]

def tree(root, stop=None, yfun=None):
  ok(root)
  yfun = yfun or (lambda r: r[root.cols.klass])
  stop = stop or the.stop or len(root.rows)**.5
  def grow(rows):
    if len(rows) <= stop: return clone(root, rows)
    best, bat, bv = BIG, None, None
    for at, col in root.cols.x.items():
      for imp, v in cuts(at, col, rows, yfun):
        if imp < best: best, bat, bv = imp, at, v
    if bat is None: return clone(root, rows)
    bcol, yes, no = root.cols.x[bat], [], []
    for r in rows:
      (yes if cutgo(bcol, r[bat], bv) else no).append(r)
    if not (yes and no): return clone(root, rows)
    return o(it=tree, at=bat, cut=bv,
             left=grow(yes), right=grow(no))

  return grow(root.rows)

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
