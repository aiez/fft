# vim: ts=2 sw=2 sts=2 et :
# knobs  only; all targets live in $(KONFIG)/Makefile
KONFIG ?= ../konfig

APP   := semble
MAIN  := semble.py
EXT   := py
LANG  := python
LINT  := ruff check *.py
TOOLS := python3:run-py ruff:lint
PKG   := python3 gawk ruff neovim tmux

# loud failure if konfig not cloned (include resolves at parse time)
$(KONFIG)/Makefile:
	@test -f $@ || { echo "missing konfig: git clone http://tiny.cc/konfig $(KONFIG)"; exit 1; }
include $(KONFIG)/Makefile

# ---- ensemble eval -------------------------------------------------
# 160-line per-tree score table (.txt) + full/sha/race compare (.csv).
# pretty: column -s, -t ~/tmp/semble_ens.csv
TMP   ?= $(HOME)/tmp
DATA  ?= ../fairnez/adult.csv
SETS  := adult bank communities compas german law

$(TMP)/semble_ens.txt: eval.py fft.py
	@mkdir -p $(TMP)
	python3 -B eval.py -f $(DATA) -t 100 -r 10 > $@
	@echo "wrote $@"

$(TMP)/semble_ens.csv: eval.py fft.py
	@mkdir -p $(TMP)
	python3 -B eval.py --csvhead > $@
	@for s in $(SETS); do \
	  python3 -B eval.py -f ../fairnez/$$s.csv -t 100 -r 10 --csv >> $@; \
	done
	@echo "wrote $@; pretty: column -s, -t $@"

.PHONY: ens
ens: $(TMP)/semble_ens.txt $(TMP)/semble_ens.csv

# best-tree-per-dataset: race on train, report goals on test. one row/set.
# pretty: column -s, -t ~/tmp/semble_best.csv
$(TMP)/semble_best.csv: eval.py fft.py
	@mkdir -p $(TMP)
	python3 -B eval.py --finalhead > $@
	@for s in $(SETS); do \
	  python3 -B eval.py -f ../fairnez/$$s.csv -t 100 -r 10 --final >> $@; \
	done
	@echo "wrote $@; pretty: column -s, -t $@"

.PHONY: best
best: $(TMP)/semble_best.csv
