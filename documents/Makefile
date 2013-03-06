LATEX       = pdflatex
CHECK_RERUN = grep Rerun $*.log

all: for_Graves.pdf

%.pdf: %.tex
	${LATEX} $<
	( ${CHECK_RERUN} && ${LATEX} $< ) || echo "Done."
	( ${CHECK_RERUN} && ${LATEX} $< ) || echo "Done."
