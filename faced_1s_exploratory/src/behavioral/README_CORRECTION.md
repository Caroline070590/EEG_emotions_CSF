# Corrected prediction analysis

The complete historical behavioral generator in this folder retains the original association and RSA workflow. Its leave-one-video-out prediction outcome normalization is superseded for Appendix J.4.

Use `revision_2026-10-25/corrections/analyses/behavioral/correct_prediction.py` from the repository root for the corrected prediction analysis. It fits each participant's outcome mean and scale using the 27 training videos only, then predicts the held-out video and reports original-unit and training-SD errors. Corrected results and portable validation accompany that script. Do not substitute the old prediction tables for the corrected J4 table or current S28 panel.
