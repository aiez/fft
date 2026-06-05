#!/usr/bin/env python3 -B
"""
semble.py, regression tree over incremental column stats (library)
(c) 2025, Tim Menzies <timm@ieee.org>, MIT license

Generic services: Data, disty, tree (min-variance cut), treeShow
(printer), confused (pd/pf/.. from (want,got) pairs), cli/the.

Columns are structs: a Num holds incremental n,mu,m2,sd (+goal);
a Sym holds n + has{value:count}. Dispatch on c.it. norm maps a
Num value to 0..1 via a logistic of its z-score (~normal CDF),
so no lo/hi to track and outliers are squashed, not stretched.

Options:
 -b --bins  numeric bins      bins=7
 -B --Budget train labels     Budget=50
 -C --Check  test labels      Check=5
 -s --seed  random seed       seed=1234567891
 -p --p     distance exponent p=2
 -R --Round repr decimals     Round=2
 -S --stop  leaf size         stop=None
 -t --trees forest size       trees=100
 -m --mtry  cols per split     mtry=auto
 -r --rows  resample mode      rows=boot
 -T --Train train fraction     Train=0.8
 -P --pos   positive klass label (auto: minority) pos=auto
 -f --file  data file
            file=/Users/timm/gits/moot/optimize/misc/auto93.csv

eg: python3 ample.py -f FILE -B 50 --tree
"""
import sys, re
from math import inf, log2, exp
from random import sample, seed, choice

BIG  = inf
TINY = 1e-30

# ## constructors -----------------------------------------------
def Num(txt="", at=0):
  return o(it=Num, at=at, txt=txt, n=0, mu=0, m2=0, sd=0,
           goal=txt[-1:]=="+")

def Sym(txt="", at=0):
  return o(it=Sym, at=at, txt=txt, n=0, has={})

def Data(src):
  src = iter(src)
  return adds(src, o(it=Data, cols=Cols(next(src)), rows=[]))

def Cols(names):
  all, x, y, klass, protected = [], [], [], None, []
  for at, s in enumerate(names):
    all += [col := (Num if s[0].isupper() else Sym)(s,at)]
    z = s.rstrip("~")[-1:]              # peel '~' so suffix detection still works
    if   z in "-+!" : y += [col]
    elif z != "X"   : x += [col]
    if z == "!"     : klass = col
    if s.endswith("~"): protected += [col]
  return o(all=all, names=names, x=x, y=y, protected=protected,
           klass=klass or (y[0] if y else None))

def Tree(): pass                     # interior tag (leaf = Data)

# ## add (one polymorphic) --------------------------------------
def add(i, v):
  if i.it is Data:
    i.rows += [v]
    for col in i.cols.all: add(col, v[col.at])
  elif v != "?":
    i.n += 1
    if i.it is Sym: i.has[v] = i.has.get(v, 0) + 1
    else:
      d     = v - i.mu
      i.mu += d / i.n
      i.m2 += d * (v - i.mu)
      i.sd  = 0 if i.n < 2 else (i.m2/(i.n-1))**.5
  return v

def adds(src, i=None):
  i = i or Num()
  for row in src: add(i, row)
  return i

def clone(root, rows=[]):
  return Data([root.cols.names] + rows)

# ## merge (s=+1 add, s=-1 subtract) ----------------------------
def merge(a, b, s=1):                # parallel Welford on two Nums
  n = a.n + s*b.n
  if n <= 0: return Num(a.txt, a.at)
  mu = (a.n*a.mu + s*b.n*b.mu) / n
  d  = b.mu - a.mu
  c  = Num(a.txt, a.at)
  c.n, c.mu = n, mu
  c.m2 = max(0, a.m2 + s*b.m2 + s*d*d*a.n*b.n / n)
  c.sd = (c.m2/(c.n-1))**.5 if c.n > 1 else 0
  return c

# ## methods ----------------------------------------------------
def mid(c):                          # central tendency
  return c.mu if c.it is Num else max(c.has, key=c.has.get)

def spread(c):                       # sd (Num) | entropy bits (Sym)
  if c.it is Num: return c.sd
  N = sum(c.has.values()) or 1
  return -sum(v/N*log2(v/N) for v in c.has.values() if v)

# ## metrics ----------------------------------------------------
def norm(c, v):                      # 0..1 logistic of z (~normal CDF)
  if v == "?": return v
  z = (v - c.mu) / (c.sd + TINY)
  return 1 / (1 + exp(-1.7 * max(-3, min(3, z))))

def bin(c, v):
  b = the.bins
  return v if c.it is Sym else min(b - 1, int(b * norm(c, v)))

def cutgo(c, v, cut):                
  return True if v=="?" else v==cut if c.it is Sym else v <= cut

def disty(data, row):                # Chebyshev-ish dist to goals
  s, n, p = 0, 0, the.p
  for c in data.cols.y:
    if c.it is Num:
      n += 1; s += abs(norm(c, row[c.at]) - c.goal)**p
  return (s/n)**(1/p) if n else 0

# ## tree (min child-variance cut) ------------------------------
# bin rows per x-col into a Num of yfun(row), then sweep for the
# min child-variance cut. total = merge(all bins); each split's
# R = merge(total, L, -1) (subtractive un-merge); impurity =
# L.m2 + R.m2. cut value = largest REAL x going left (hi[k]).
def cuts(col, rows, yfun):           # yield (impurity, real cut value)
  ys, hi, total = {}, {}, Num()
  for r in rows:
    if (x := r[col.at]) != "?":
      y = add(total, yfun(r))           
      k = bin(col, x)
      b = ys[k] = ys.get(k) or Num()
      add(b, y)
      if col.it is Num: hi[k] = max(hi.get(k, x), x)
  yield from cutsBinary(sorted(ys), ys, hi, total, col)

def cutsBinary(ks, ys, hi, total, col):
  if not ks: return None
  if col.it is Sym:
    for k in ks:
      R = merge(total, ys[k], -1)
      if ys[k].n and R.n: yield ys[k].m2 + R.m2, k
  else:                             
    L = Num()
    for k in ks[:-1]:
      L = merge(L, ys[k])
      R = merge(total, L, -1)
      if L.n and R.n: yield L.m2 + R.m2, hi[k]

def tree(root, stop=None, yfun=None, mtry=None):
  yfun = yfun or (lambda r: r[root.cols.klass.at])
  stop = stop or the.stop or len(root.rows)**.5
  def grow(rows):
    if len(rows) > stop:
      best, bcol, bv = BIG, None, None
      xs = root.cols.x               # mtry None = all x-cols (single tree)
      if mtry and mtry < len(xs): xs = sample(xs, mtry)  # random subspace
      for col in xs:                 # best cut over candidate x-cols
        for imp, v in cuts(col, rows, yfun):
          if imp < best: best, bcol, bv = imp, col, v
      if bcol is not None:
        yes, no = [], []
        for r in rows:
          (yes if cutgo(bcol, r[bcol.at], bv) else no).append(r)
        if yes and no:
          return o(it=Tree, at=bcol.at, cut=bv,
                   left=grow(yes), right=grow(no))
    return clone(root, rows)         # one leaf path
  return grow(root.rows)

def leaves(t):                       # yield the leaf Datas
  if t.it is Data: yield t
  else: yield from leaves(t.left); yield from leaves(t.right)

def treeLeaf(root, t, row):          # route a row to its leaf
  while t.it is Tree:
    c = root.cols.all[t.at]
    t = t.left if cutgo(c, row[t.at], t.cut) else t.right
  return t

# ## forest (vary inductive bias: random rows + random cols) ----
# 100 trees, each on a resampled train set (bootstrap | balanced |
# all) grown over a random mtry-col subspace per split. predict =
# majority klass vote across trees. diversity from rows+cols only;
# each tree still takes the GREEDY best cut for its candidate cols.
def resample(rows, klass, mode):     # row diversity per tree
  if mode == "boot": return [choice(rows) for _ in rows]    # bootstrap
  if mode == "bal":                  # balanced: undersample to min class
    by = {}
    for r in rows: by.setdefault(r[klass.at], []).append(r)
    m = min(len(v) for v in by.values())
    return [choice(v) for v in by.values() for _ in range(m)]
  return rows[:]                     # "all": col diversity only

def forest(data, train, pos, n=None, mtry=None, mode=None):
  n     = n or the.trees
  mode  = mode or the.rows
  xs    = data.cols.x
  mtry  = mtry or (the.mtry if isinstance(the.mtry, int)
                   else max(1, int(len(xs)**.5)))           # auto = sqrt(p)
  klass = data.cols.klass
  yfun  = lambda r: 1.0 if r[klass.at] == pos else 0.0      # numeric encode
  trees = []
  for _ in range(n):
    root = clone(data, resample(train, klass, mode))
    trees.append((root, tree(root, yfun=yfun, mtry=mtry)))
  return trees

def predicts(root, t, rows):         # (want, got) pairs for confused()
  return [(r[root.cols.klass.at],
           mid(treeLeaf(root, t, r).cols.klass)) for r in rows]

def treeMetrics(root, t, val, pos):  # 6 goals: pd,pf,prec, mean spd/ard/aaod
  m  = confused(predicts(root, t, val)).get(pos) or o(pd=0, pf=1, prec=0)
  fs = list(scoreFairness(root, t, val, pos).values())   # per protected col
  avg = lambda k: sum(f[k] for f in fs)/len(fs) if fs else 0
  return [m.pd, m.pf, m.prec, avg("spd"), avg("ard"), avg("aaod")]

# score each bias on 6 goals, then disty (Chebyshev-to-heaven over
# norm'd goals) picks the tree closest to the ideal corner.
SCORES = ["idX", "Pd+", "Pf-", "Prec+", "Spd-", "Ard-", "Aaod-"]
def forestSelect(trees, val, pos):   # champion = min distance-to-heaven
  rows  = [[i] + treeMetrics(rt[0], rt[1], val, pos)
           for i, rt in enumerate(trees)]
  board = Data([SCORES] + rows)
  best  = min(board.rows, key=lambda r: disty(board, r))
  return trees[int(best[0])], board, best

def split(data, frac=None):          # stratified train/test by klass
  frac, at, by = frac or the.Train, data.cols.klass.at, {}
  for r in data.rows: by.setdefault(r[at], []).append(r)
  tr, te = [], []
  for v in by.values():
    v = sample(v, len(v))            # shuffle within class
    k = int(frac * len(v))
    tr += v[:k]; te += v[k:]
  return tr, te

def treeShow(root, t):               # -> matrix of str (use print2d())
  ys     = [c for c in root.cols.y if c.it is Num]
  rows   = lambda t: t.rows if t.it is Data else rows(t.left)+rows(t.right)
  win    = wins(root)
  wmid   = lambda rs: sum(win(r) for r in rs)/max(1, len(rs))
  out    = [["", "win", *[c.txt for c in ys], "n", "tree"]]
  leaves = []
  def walk(t, lvl, label):
    rs = rows(t); d = clone(root, rs)
    out.append(["", "%d" % round(100 * wmid(rs)),
                *["%.1f" % d.cols.all[c.at].mu for c in ys],
                str(len(rs)), "|  "*(lvl-1) + label])
    if t.it is Tree:
      c  = root.cols.all[t.at]
      op = ("<=", ">") if c.it is Num else ("==", "!=")
      walk(t.left,  lvl+1, "%s %s %s" % (c.txt, op[0], t.cut))
      walk(t.right, lvl+1, "%s %s %s" % (c.txt, op[1], t.cut))
    else:
      leaves.append(len(out)-1)
  walk(t, 0, "")
  leaves.sort(key=lambda i: float(out[i][1]))
  out[leaves[0]][0], out[leaves[-1]][0] = "-", "+"
  return out

# ## acquire (active-learning eval) -----------------------------
# split half/half; `select` picks Budget rows of train to label
# (default: random); grow a disty-tree on them; sort test by the
# leaf disty its rows route to; label the top Check; the best of
# those = `found`. WIN = how far found beats the mean toward best:
# 100 = hit the optimum, 0 = no better than average.
def wins(data):                  # row -> 0..1 (1 = optimum, 0 = mean)
  ys = sorted(disty(data, r) for r in data.rows)
  lo, mu = ys[0], sum(ys)/len(ys)
  return lambda row: 1 - (disty(data, row) - lo)/(mu - lo + TINY)

def generalized(data, select=None):
  rows = sample(data.rows, len(data.rows))      # shuffle
  win  = wins(data)                            
  select = select or (lambda rs: sample(rs, min(the.Budget, len(rs))))
  half = len(rows)//2
  train, test = rows[:half], rows[half:]
  root = clone(data, select(train))             # labeled budget
  t    = tree(root, yfun=lambda r: disty(data, r))
  for lf in leaves(t):                           # cache leaf disty-mean
    lf._mu = sum(disty(data, r) for r in lf.rows)/max(1, len(lf.rows))
  test.sort(key=lambda r: treeLeaf(root, t, r)._mu)
  return int(100 * max(win(r) for r in test[:the.Check]))

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

# ## fairness (group gaps from confused() output) --------------
# a, b = confused(pairs) of two protected groups; pos = positive label.
# All return 0 at perfect parity; higher = more unfair.
def rate(m):                      # acceptance rate: P(Ŷ=m.label) in group
  return (m.tp + m.fp) / (m.tp + m.fp + m.fn + m.tn + 1e-32)

def spd(a, b, pos):               # statistical parity diff: ΔP(Ŷ=pos)
  return abs(rate(a[pos]) - rate(b[pos]))

def ard(a, b, pos):               # accuracy rate diff: pooled-class acc gap
  return abs(a[pos].acc - b[pos].acc)

def aaod(a, b, pos):              # avg abs odds diff: ½(|ΔFPR| + |ΔTPR|)
  return 0.5 * (abs(a[pos].pf - b[pos].pf) + abs(a[pos].pd - b[pos].pd))

def groups(col, rows):            # top2 syms (Sym) or median split (Num)
  if col.it is Sym:
    g1, g2 = sorted(col.has, key=col.has.get, reverse=True)[:2]
    return ([r for r in rows if r[col.at] == g1], g1,
            [r for r in rows if r[col.at] == g2], g2)
  med = col.mu
  return ([r for r in rows if r[col.at] != "?" and r[col.at] <= med], "lo",
          [r for r in rows if r[col.at] != "?" and r[col.at] >  med], "hi")

def scoreFairness(root, t, rows, pos):   # -> {col.txt: o(g1,g2,spd,ard,aaod)}
  klass = root.cols.klass
  pred  = {}                                # row id -> predicted label
  for leaf in leaves(t):
    pk = mid(leaf.cols.klass)
    for r in leaf.rows: pred[id(r)] = pk
  out = {}
  for col in root.cols.protected:
    s1, g1, s2, g2 = groups(col, rows)
    if not s1 or not s2: continue
    a = confused([(r[klass.at], pred[id(r)]) for r in s1 if id(r) in pred])
    b = confused([(r[klass.at], pred[id(r)]) for r in s2 if id(r) in pred])
    if pos not in a or pos not in b: continue
    out[col.txt] = o(g1=g1, g2=g2, n1=len(s1), n2=len(s2),
                     spd=spd(a,b,pos), ard=ard(a,b,pos), aaod=aaod(a,b,pos))
  return out

# ## lib -------------------------------------------------------
def print2d(rows, sep="  "):      # rjust all but the last col (left, raw)
  rows = [[str(x) for x in r] for r in rows]
  ws = [max(len(r[c]) for r in rows if c < len(r))
        for c in range(max(len(r) for r in rows))]
  for r in rows:
    print(sep.join([r[c].rjust(ws[c]) 
                    for c in range(len(r)-1)] + [r[-1]]))

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
  out = [["col", "n", "mid", "spread"]]
  for c in d.cols.all:
    m = round(mid(c), the.Round) if c.it is Num else mid(c)
    out += [[c.txt, c.n, m, round(spread(c), the.Round)]]
  print2d(out)

def test_confused():            # confusion stats from toy (want,got)
  pairs = ([("y","y")]*8 + [("y","n")]*2 +
           [("n","n")]*7 + [("n","y")]*3)
  out = [["label","tp","fp","fn","tn","pd","pf","prec","acc","f1"]]
  for k, m in confused(pairs).items():
    out += [[k, m.tp, m.fp, m.fn, m.tn] +
            [round(m[x], 2) for x in ("pd","pf","prec","acc","f1")]]
  print2d(out)

def _lookupPos(path):           # read labels.py next to the CSV (if any)
  import importlib.util, os
  lab = os.path.join(os.path.dirname(os.path.abspath(path)), "labels.py")
  if not os.path.exists(lab): return None
  spec = importlib.util.spec_from_file_location("labels", lab)
  mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
  return getattr(mod, "POSITIVE", {}).get(os.path.basename(path))

def test_fairness():            # SPD/ARD/AAOD per protected col of -f
  d = Data(csv(the.file))
  if not d.cols.protected:
    print("no protected cols (need ~ suffix); see ../fairnez/protect.py")
    return
  pos = the.pos
  if pos in (None, "", "auto"):
    pos = _lookupPos(the.file)
    if pos is None:
      pos = min(d.cols.klass.has, key=d.cols.klass.has.get)
  klass_at = d.cols.klass.at
  yfun = lambda r: 1.0 if r[klass_at] == pos else 0.0     # numeric encoding
  t = tree(d, yfun=yfun)
  out = [["protected", "g1", "n1", "g2", "n2", "SPD", "ARD", "AAOD"]]
  for txt, m in scoreFairness(d, t, d.rows, pos).items():
    out += [[txt, m.g1, m.n1, m.g2, m.n2,
             round(m.spd, 3), round(m.ard, 3), round(m.aaod, 3)]]
  print(the.file.split("/")[-1], "  positive label =", pos)
  print2d(out)

def test_general():             # mean WIN over 20 generalized runs
  d  = Data(csv(the.file))
  ws = adds((generalized(d) for _ in range(20)))
  print(round(ws.mu), len(d.rows), len(d.cols.x), len(d.cols.y),
        the.Budget, the.Check, the.file.split("/")[-1])

def test_forest():              # 100 biases -> pick champion on val -> test
  d     = Data(csv(the.file))
  klass = d.cols.klass
  pos   = the.pos
  if pos in (None, "", "auto"):
    pos = min(klass.has, key=klass.has.get)        # minority label
  yfun     = lambda r: 1.0 if r[klass.at] == pos else 0.0
  tr,  te  = split(d)                              # 80:20 train:test
  fit, val = split(clone(d, tr))                   # train -> fit:val
  trees    = forest(d, fit, pos)                   # 100 varied biases
  champ, board, win = forestSelect(trees, val, pos)  # closest-to-heaven bias
  base     = (root := clone(d, tr), tree(root, yfun=yfun))  # single-tree
  out = [["model", "label", "pd", "pf", "prec", "acc", "f1"]]
  for name, (rt, t) in [("champ", champ), ("tree", base)]:
    if m := confused(predicts(rt, t, te)).get(pos):
      out += [[name, pos] + [round(m[x], 2)
              for x in ("pd","pf","prec","acc","f1")]]
  print(the.file.split("/")[-1],
        " biases=%s mtry=%s rows=%s pos=%s d2heaven=%.2f n_test=%d" %
        (the.trees, the.mtry, the.rows, pos, disty(board, win), len(te)))
  print2d(out)

def test_tree():                # tree on N rows of -f
  head, *body = csv(the.file)
  for _ in range(100):
    rows = sample(body, min(the.Budget, len(body)))
    d = Data([head] + rows)
    t = tree(d)
  print2d(treeShow(d,t))

# ## main -------------------------------------------------------
the = settings(__doc__)
seed(the.seed)
if __name__ == "__main__":
  cli(the, __doc__, globals())
