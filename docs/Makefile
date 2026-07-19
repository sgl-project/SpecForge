# Minimal Makefile for Sphinx documentation
SPHINXOPTS    ?=
SPHINXBUILD   ?= sphinx-build
SPHINXAUTOBUILD ?= sphinx-autobuild
SOURCEDIR     = .
BUILDDIR      = _build
PORT          ?= 8003

help:
	@$(SPHINXBUILD) -M help "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)
	@echo ""
	@echo "Additional targets:"
	@echo "  serve       to build and serve documentation with auto-build and live reload"

# Serve documentation with auto-build and live reload
serve:
	@echo "Starting auto-build server at http://0.0.0.0:$(PORT)"
	@$(SPHINXAUTOBUILD) "$(SOURCEDIR)" "$(BUILDDIR)/html" \
		--host 0.0.0.0 \
		--port $(PORT) \
		--watch $(SOURCEDIR) \
		--re-ignore ".*\.(pyc|pyo|pyd|git)"

.PHONY: help Makefile clean serve

%: Makefile
	@$(SPHINXBUILD) -M $@ "$(SOURCEDIR)" "$(BUILDDIR)" $(SPHINXOPTS) $(O)

clean:
	rm -rf $(BUILDDIR)
