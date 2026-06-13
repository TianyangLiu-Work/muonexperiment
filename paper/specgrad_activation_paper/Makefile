.PHONY: all two-page clean

all:
	pdflatex main.tex
	bibtex main
	pdflatex main.tex
	pdflatex main.tex

two-page:
	xelatex two_page.tex
	xelatex two_page.tex

clean:
	rm -f *.aux *.bbl *.blg *.log *.out *.toc *.fls *.fdb_latexmk *.synctex.gz
