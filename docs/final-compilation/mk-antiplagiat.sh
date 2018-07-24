#!/bin/sh

pdftk title-page.pdf specification.pdf annotation.pdf output header.pdf
pdftk thesis.pdf cat r2-end output bib.pdf

convert -density 600 +antialias header.pdf header-img.pdf
convert -density 600 +antialias bib.pdf bib-img.pdf

pdftk H=header-img.pdf T=thesis.pdf L=bib-img.pdf cat H T3-r3 L output thesis-zabelkin.pdf

rm header.pdf bib.pdf bib-img.pdf header-img.pdf