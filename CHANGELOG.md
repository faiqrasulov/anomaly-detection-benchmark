# Changelog

Notable changes to this project, newest first.

## 0.1.0 (2026-11-17)

First tagged release. It contains everything below.

## 2026-11-16

- Package versions pinned in `requirements.lock`, with a `make install-locked` target.
- README: the One-Class SVM is a Nystroem + SGD approximation and is the one detector that does not reproduce to the digit.

## 2026-11-15

- First version: 8 unsupervised detectors on KDD Cup 99 and Forest CoverType, scored by precision-recall, seed stability, runtime and alert budgets.
- README: three of the ranked datasets are nested KDD subsets, so the rank test's independence assumption does not hold.
