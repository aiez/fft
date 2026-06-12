<!-- Copyright (c) 2026 Tim Menzies, MIT License https://opensource.org/licenses/MIT -->
<img align="right" src="https://img.shields.io/badge/Purpose-AI·Applications·Teaching-7b68ee?logo=githubcopilot&logoColor=white" alt="Purpose"> <a href="https://timm.fyi"> <img align="right" src="https://img.shields.io/badge/Author-timm-dc143c?logo=readme&logoColor=white" alt="Author"></a> <img align="right" src="https://img.shields.io/badge/Language-Python-000080?logo=python&logoColor=white" alt="Language"><a href="https://choosealicense.com/licenses/mit/"> <img align="right" src="https://img.shields.io/badge/License-MIT-32cd32?logo=open-source-initiative&logoColor=white" alt="License"></a> 

### [http://tiny.cc/fft](http://tiny.cc/fft)
A fast-and-frugal tree (FFT) makes urgent binary choices via a few yes/no questions, deliberately ignoring most of the data: fast since little computation, frugal since it often stops after one or two cues. Here: `fft.py` is the core (one pass = 16 candidate trees, one per bias string, from min-variance cuts on incremental Welford stats; ~240 lines, pure Python, no deps); `eval.py` samples many such trees and picks winners cheaply (successive halving, Hoeffding racing), scoring accuracy plus fairness. Data is CSV; column-name suffix tags type and goal.

```bash
# install and test
git clone http://tiny.cc/optimiz && git clone http://tiny.cc/fft fft
cd fft && python3 -B fft.py -f ../optimiz/auto93.csv
```

**Sections:** [NAME](#name) | [SYNOPSIS](#synopsis) | [OPTIONS](#options) | [DATA](#data) | [TESTS](#tests) | [TREE OUTPUT](#tree-output) | [EXIT](#exit) | [SEE ALSO](#see-also) | [LICENSE](#license) | [AUTHOR](#author)

**Files:** [fft.py](#file-fft-py) | [eval.py](#file-eval-py)

## NAME

    fft - fast-frugal multi-objective trees; eval - sample + race them

## SYNOPSIS

    python3 -B fft.py  [-flag VAL]... [--grows|--trees]
    python3 -B eval.py [-flag VAL]... [--sha|--csv|--final]

## OPTIONS

    fft.py (defaults from its docstring; SSOT):

      -s seed     random seed              (1234567891)
      -p dist     distance exponent        (2)
      -b bins     numeric bin count        (7)
      -d depth    max tree depth           (4)
      -R Round    display decimals         (2)
      -f file     data file                ($DOOT/optimiz/auto93.csv)

    eval.py:

      -f file     data file                ($DOOT/fairnez/adult.csv)
      -t trainN   train rows per sample    (200; 100 in --sha/--final)
      -P pos      positive label           (auto: labels.py, else minority)
      -r repeats  resamples of trainN rows (10)
      -S start    first rung, rows         (50; --sha/--final only)

## DATA

    CSV with header row. Each column name encodes type + role
    via first char (case) and last char (suffix):

      first char UPPER  -> numeric (Num)
      first char lower  -> symbolic (Sym)
      suffix '+'        -> numeric goal, maximize
      suffix '-'        -> numeric goal, minimize
      suffix '!'        -> symbolic goal (klass)
      suffix '~'        -> protected attribute (fairness)
      suffix 'X'        -> ignore
      else              -> predictor

    Missing values: '?'. Example header:

      Clndrs,Volume,HpX,Model,origin,Lbs-,Acc+,Mpg+

## TESTS

    fft.py (multi-goal regression to distance-to-heaven):

      (none)      grow 16 trees, show the lowest-error one
      --trees     show all 16 candidate trees + their bias + error
      --grows     timing: trees/second over repeated samples

    eval.py (binary classification + fairness, e.g. fairnez CSVs):

      (none)      table: every tree x (pd, pf, prec, spd...) by disty
      --sha       full grid vs successive-halving vs Hoeffding race
      --csv       same comparison, one CSV row (for make ens)
      --final     race on train, report winner on test (make best)

## TREE OUTPUT

    fft.py prints one decision spine: each line is a cue; rows
    matching it stop there (their distance-to-heaven d2h, 0=best),
    all others fall through to the next cue.

      if Volume <= 83        then d2h 0.28 n=18
      if Model <= 74         then d2h 0.65 n=140
      ...
                             leaf d2h 0.66 n=99

    Example:

      python3 -B fft.py -f ../optimiz/config_SS-N.csv --trees

## EXIT

    0  success
    1  bad file / flag

## SEE ALSO

    http://tiny.cc/fairnez  fairness CSVs scored by eval.py
    http://tiny.cc/optimiz  example CSVs (auto93, config_SS-N, ...)
    http://tiny.cc/konfig   shared Makefile, bashrc, nvim, tmux

## LICENSE

    MIT.  https://choosealicense.com/licenses/mit/
    (c) 2025 Tim Menzies.

## AUTHOR

    Tim Menzies <timm@ieee.org>
