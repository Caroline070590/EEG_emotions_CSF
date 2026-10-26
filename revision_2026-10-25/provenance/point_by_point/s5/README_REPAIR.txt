Supplementary Figure S5 label-only repair

The source docstring incorrectly described the absolute-correlation distance as nonmetric. The manuscript proof correctly establishes a sign-quotient pseudometric. The two panel-c axes now name absolute-correlation and signed-correlation distances.

Run pdflatex label_overlay.tex in this folder to reproduce the vector-label replacement. Original_S5.pdf is the unchanged scientific figure used as input (actual filename: original_S5.pdf). The 0.0023 bp vertical placement correction compensates the PDF inclusion origin rounding and restores exact pixel positions. No scientific panels were regenerated.

At 300 and 600 dpi the output is pixel-identical to the original outside the two documented axis-label rectangles. Source AST comparison confirms that only module docstrings and two label strings changed; all calculations and all_contrasts.csv are unchanged. Invisible old labels can remain in plain-text extraction because the input PDF is preserved beneath white vector patches.
