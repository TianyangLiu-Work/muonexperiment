.PHONY: all main two-page tectonic main-tectonic two-page-tectonic clean

PDFLATEX ?= pdflatex
BIBTEX ?= bibtex
XELATEX ?= xelatex
TECTONIC ?= tectonic

all:
	@if command -v "$(PDFLATEX)" >/dev/null 2>&1 && command -v "$(XELATEX)" >/dev/null 2>&1; then \
		$(MAKE) main two-page; \
	elif command -v "$(TECTONIC)" >/dev/null 2>&1; then \
		$(MAKE) tectonic; \
	else \
		echo "No LaTeX engine found. Install pdflatex/xelatex or tectonic."; \
		exit 1; \
	fi

main:
	$(PDFLATEX) main.tex
	$(BIBTEX) main
	$(PDFLATEX) main.tex
	$(PDFLATEX) main.tex

two-page:
	$(XELATEX) two_page.tex
	$(XELATEX) two_page.tex

tectonic: main-tectonic two-page-tectonic

main-tectonic:
	$(TECTONIC) main.tex

two-page-tectonic:
	$(TECTONIC) two_page.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz
