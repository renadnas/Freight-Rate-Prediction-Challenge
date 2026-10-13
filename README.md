# Freight-Rate-Prediction-Challenge
The task is to predict the posted rate of 12,000 loads dated November and December 2025, using 48,000 labeled loads from January to October 2025. Because the loads to predict lie entirely in the future, the main design decision was how to validate: I used time-based, expanding-window validation instead of a random split.

# Freight Rate Prediction

Predicts the posted rate of freight loads. The model is trained on 48,000 labeled loads (Jan to Oct 2025) and predicts 12,000 unlabeled loads (Nov to Dec 2025).

**Model:** LightGBM trained on log(price).
**Validation:** time-based, expanding-window folds (train on the past, test on the next two months).

## Results (time-based cross-validation, average of 3 folds)

| Model | MAE ($) | MAPE (%) | MedianAPE (%) |
|---|---|---|---|
| Baseline (distance x typical price per mile) | 219.35 | 10.53 | 6.85 |
| Linear regression (log price) | 135.52 | 5.88 | 3.30 |
| **LightGBM (log price)** | **111.36** | **4.84** | **2.07** |

All rows are scored, including the ~1.4% of labels flagged as price outliers. Excluding them, LightGBM reaches 2.54% MAPE. A random split would have reported a misleading 1.16% MedianAPE for the same model. Details are in `report/Freight_Rate_Report.docx`.

## Repository layout

```
.
├── freight_rate_prediction.ipynb   the whole pipeline, top to bottom
├── score.py                        provided scorer (unchanged)
├── requirements.txt
├── validation_predictions.csv      final predictions: load_id,predicted_rate
├── december_predictions.csv        predictions for the fixed December chart
├── scorer_results/candidate_december.png
├── model_comparison.png
└── report/Freight_Rate_Report.docx
```

## How to run

Requires Python 3.10+ (tested on Python 3.12 and 3.13; results are identical across the pandas versions tried, 2.2 and 2.3).

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
python -m pip install -r requirements.txt

# run the whole notebook (about 80 seconds) ...
jupyter nbconvert --to notebook --execute --inplace freight_rate_prediction.ipynb

# ... then validate the outputs with the provided scorer
python score.py --predictions validation_predictions.csv --december-predictions december_predictions.csv
```

You can also open `freight_rate_prediction.ipynb` in Jupyter or Google Colab and choose *Run all*. Keep `data/`, `score.py` and `requirements.txt` in the same folder as the notebook (in Colab, upload them to the working folder, next to the notebook).

If `data/validation_predictions_template.csv` is present, the notebook also checks that the IDs match it.

## What the notebook does

1. **Clean.** Negative weights flipped to positive; missing weight filled with the median for the equipment type; missing market index filled with the same-date median; missing flags kept. About 1.4% of training prices are far from what distance and equipment imply (robust score above 5). They are flagged, excluded from training only, and kept in every test fold.
2. **Features.** Distance (raw and log), equipment, weight, pickup/delivery coordinates and trip direction, and the market signals (`market_index`, `quote_signal` and their daily medians). City names are not used because 8 validation pickup cities never appear in training. Month is not used because training only covers months 1 to 10.
3. **Validation.** Three expanding-window folds, each testing on a two-month future window, the same horizon as the final task. The notebook asserts that test dates are always after training dates.
4. **Model.** LightGBM on log(price). It beats the baseline and the linear model on every fold, and three different settings gave nearly identical scores.
5. **December chart.** The December file has no market signals or coordinates. Coordinates come from a city lookup; for the market signals, each date's median from `validation.csv` is used (a "typical load" for that date).

## Known limitations

* The model under-predicts slightly in every test month and gets worse the further it forecasts, so December is expected to be somewhat worse than November.
* Price outliers in the labels cannot be predicted. They are about 1.4% of rows but cause roughly half of the dollar error.
* The model has never seen a December, so it cannot learn holiday effects. The December curve follows the market signals only.
* Imputation medians and the outlier formula are computed on the full labeled set (a handful of numbers); the leakage is negligible.

All randomness is seeded (`random_state=42`).

