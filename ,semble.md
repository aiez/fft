<!-- Copyright (c) 2026 Tim Menzies, MIT License https://opensource.org/licenses/MIT -->
<img xalign="right" src="https://img.shields.io/badge/Purpose-AI·Applications·Teaching-7b68ee?logo=githubcopilot&logoColor=white" alt="Purpose"> <a href="https://timm.fyi"> <img xalign="right" src="https://img.shields.io/badge/Author-timm-dc143c?logo=readme&logoColor=white" alt="Author"></a> <img xalign="right" src="https://img.shields.io/badge/Language-Python-000080?logo=python&logoColor=white" alt="Language"><a href="https://choosealicense.com/licenses/mit/"> <img xalign="right" src="https://img.shields.io/badge/License-MIT-32cd32?logo=open-source-initiative&logoColor=white" alt="License"></a> 

### [http://tiny.cc/semble](http://tiny.cc/semble)
Regression trees by greedy min-variance cuts on incremental column stats (Welford μ/σ). ~340 lines, pure Python, no deps. Data is CSV; column-name suffix tags type and goal.

```bash
# install and test
git clone http://tiny.cc/semble && git clone http://tiny.cc/optimiz
cd semble && python3 -B semble.py -f ../optimiz/auto93.csv --tree
```

## NAME

    semble - assemble models by trying many inductive biases

## SYNOPSIS

    python3 -B semble.py [-flag VAL]... --TEST

## OPTIONS

    -b --bins     numeric bin count        (7)
    -B --Budget   train-row labels         (50)
    -C --Check    test-row labels          (5)
    -s --seed     random seed              (1234567891)
    -p --p        distance exponent        (2)
    -R --Round    display decimals         (2)
    -S --stop     min leaf size            (sqrt N rows)
    -f --file     data file                (auto93.csv)

## DATA

    CSV with header row. Each column name encodes type + role
    via first char (case) and last char (suffix):

      first char UPPER  -> numeric (Num)
      first char lower  -> symbolic (Sym)
      suffix '+'        -> numeric goal, maximize
      suffix '-'        -> numeric goal, minimize
      suffix '!'        -> symbolic goal (klass)
      suffix 'X'        -> ignore
      else              -> predictor

    Missing values: '?'. Example header:

      Clndrs,Volume,HpX,Model,origin,Lbs-,Acc+,Mpg+

## TESTS

    Each --name calls test_name() (re-seeded). Listed under
    `## egs` in source.

      --the       print current config
      --stats     mid + spread per column of -f
      --confused  pd/pf/prec/acc/f1 over toy (want,got) pairs
      --tree      regression tree on Budget rows of -f
      --general   mean WIN over 20 active-learning runs

## TREE OUTPUT

    Columns:  mark  win  GOAL±...  n  tree

      win    0..100. 100 = optimum, 0 = mean, neg = worse.
      +      best leaf
      -      worst leaf

    Example:

      python3 -B semble.py -f ../optimiz/SS-N.csv --tree

## EXIT

    0  success
    1  bad flag (usage printed)

## SEE ALSO

    http://tiny.cc/optimiz  example CSVs (auto93, SS-N, ...)
    http://tiny.cc/konfig   shared Makefile, bashrc, nvim, tmux

## LICENSE

    MIT.  https://choosealicense.com/licenses/mit/
    (c) 2025 Tim Menzies.

## AUTHOR

    Tim Menzies <timm@ieee.org>
