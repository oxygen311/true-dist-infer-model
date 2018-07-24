#!/bin/bash

xelatex thesis
biber thesis
xelatex thesis
xelatex thesis

rm thesis.{aux,log,bbl,bcf,blg,run.xml,toc,tct}
rm *.aux
