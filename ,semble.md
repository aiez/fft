<!-- Copyright (c) 2026 Tim Menzies, MIT License https://opensource.org/licenses/MIT -->
<a href="https://timm.fyi"><img align="right" src="https://img.shields.io/badge/Author-timm-dc143c?logo=readme&logoColor=white" alt="Author"></a><img align="right" src="https://img.shields.io/badge/Language-Python-000080?logo=python&logoColor=white" alt="Language"><a href="https://choosealicense.com/licenses/mit/"><img align="right" src="https://img.shields.io/badge/License-MIT-32cd32?logo=open-source-initiative&logoColor=white" alt="License"></a><img align="right" src="https://img.shields.io/badge/Purpose-AI·Applications·Teaching-7b68ee?logo=githubcopilot&logoColor=white" alt="Purpose">

### [http://tiny.cc/semble](http://tiny.cc/semble)
Regression trees by greedy min-variance cuts on incremental column stats (Welford μ/σ). ~340 lines, pure Python, no deps. Data is CSV; column-name suffix tags type and goal.

```bash
# install and test
git clone http://tiny.cc/semble && git clone http://tiny.cc/optimiz
cd semble && python3 -B semble.py -f ../optimiz/auto93.csv --tree
```

## NAME
**semble** — assemble models by trying many inductive biases.

## SYNOPSIS

    python3 -B semble.py [-flag VAL]... --TEST

## DATA
CSV with header row. Each column name encodes type + role via its first char and last char:

| suffix | meaning                         |
|--------|---------------------------------|
| `+`    | numeric goal, maximize          |
| `-`    | numeric goal, minimize          |
| `!`    | symbolic goal (klass)           |
| `X`    | ignore                          |
| (none) | predictor                       |

Names starting with an uppercase letter are numeric (`Num`), else symbolic (`Sym`). Example header:

    Clndrs,Volume,HpX,Model,origin,Lbs-,Acc+,Mpg+

Missing values: `?`.

## OPTIONS

| flag         | meaning                  | default       |
|--------------|--------------------------|---------------|
| -b --bins    | numeric bin count        | 7             |
| -B --Budget  | train-row labels         | 50            |
| -C --Check   | test-row labels          | 5             |
| -s --seed    | random seed              | 1234567891    |
| -p --p       | distance exponent        | 2             |
| -R --Round   | display decimals         | 2             |
| -S --stop    | min leaf size            | sqrt(N rows)  |
| -f --file    | data file                | auto93.csv    |

## TESTS
Each `--name` calls `test_name()` (re-seeded). To list them: read funcs under `## egs` in source.

| flag         | what                                                |
|--------------|-----------------------------------------------------|
| --the        | print current config                                |
| --stats      | mid + spread per column of `-f`                     |
| --confused   | pd/pf/prec/acc/f1 over toy (want,got) pairs         |
| --tree       | regression tree on Budget rows of `-f`              |
| --general    | mean WIN over 20 active-learning runs               |

## TREE OUTPUT
Columns: `mark  win  GOAL±...  n  tree`.
- `win` is 0..100. 100 = hit optimum, 0 = no better than mean (negative = worse).
- `+` marks best leaf, `-` marks worst leaf.

    python3 -B semble.py -f ../optimiz/SS-N.csv --tree

## EXIT
0 on success. `bad flag: ...` and usage on unknown flag.

## SEE ALSO
- `http://tiny.cc/optimiz` — example CSVs (auto93, SS-N, ...)
- `http://tiny.cc/konfig` — shared Makefile, bashrc, nvim, tmux

## LICENSE
[MIT](https://choosealicense.com/licenses/mit/) © 2025 Tim Menzies.

## AUTHOR
Tim Menzies <timm@ieee.org>.
