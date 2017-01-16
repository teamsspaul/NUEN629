#!/usr/bin/env bash
# PDF Blank Filler
#
# (Creative Commons 0)
# To the extent possible under law,
# I have waived all copyright and related
# or neighboring rights to PDF Blank Filler.
#
################################################################################
#
# PDFs that have been generated with white text to hide content still contain
# that text---it's just white.
#
# This short script replaces all white content's color with red in a new file
# whose name is prefixed with "filled-"
#
################################################################################
#
# Requires:
# sed   - stream editor, typically included in linux installs
# perl  - perl programming language
# pdftk - pdf toolkit, typically available in your software repository
#
# This will only be successful on pdfs that include text and not images of text.
# Try highlighting all text in a pdf viewer, if you see hidden text get
# highlighted it will likely work on it.
#mkdir filled
for f in *.pdf; do
  #pdftk $f output "filled/tmp-$f" uncompress

  # Dr. Adam's
  #sed -i "s/1 1 1 rg/1 0 0 rg/g" "filled/tmp-$f" # RGB
  #sed -i "s/1 1 1 RG/1 0 0 RG/g" "filled/tmp-$f" # RGB
  
  # Dr. McClarren's
  perl -0777 -i -pe "s/0\s0\s0\s0\sk/0 1 1 0 k/igs" "filled/tmp-$f" # CYMK
  perl -0777 -i -pe "s/0\s0\s0\s0\sK/0 1 1 0 K/igs" "filled/tmp-$f" # CYMK


  #pdftk "filled/tmp-$f" output "filled/filled-$f" compress
  #rm "filled/tmp-$f"
#  qpdf --qdf --object-streams=disable "filled/filled-$f" "filled/uncomp-$f"
done
