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
