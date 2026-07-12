# Chicago Neighborhood Premium — Project Summary

Chicago has 77 officially recognized community areas, and even adjacent ones can differ radically in character and price. This project quantifies the premium a homebuyer pays for the neighborhood itself. The part of a single-family home's price attributable to where it is, separate from what the house physically is. The result is a dollar figure per community area, useful both to builders deciding where construction is most profitable and as an indicator of neighborhood quality not captured by property features. 

## Executive process summary

**1. Data assembly.** Many potential data sources were explored including Zillow research data, chicago health atlas, and university of Chicago. We finally decided on the Cook county open data portal (datacatalog.cookcountyil.gov). This source contains enourmous data for Cook county houses, starting from 1999. Four datasets were downloaded from the portal, cleaned, and merged on 14 digit PIN identifier: parcel sales (the target variable), improvement characteristics (square footage, age, rooms, construction), parcel universe (geography, community area, walkability), and parcel proximity (distances to amenities). Years 2021-2025 were selected partially for reducing data size and also because data records before 2021 were unreliable due to a change in tax laws. In datasets that listed the same parcel for multiple years, the most recent entry compared to the sale year was selected. Single family houses were selected for the first analysis. Out of more than 200 features available in the four datasets, about 70 most relevant were selected. City of Chicago homicide data (yearly average for each neighbourhood) was joined in as a crime signal. The merged dataset lives in `Data/sale_and_crime_data/`. 

**2. Target validation and feature selection.** Sale price and tax-assessed value were compared and found to disagree enough that our project committed to sale price alone. Features encoding *relative* location (latitude/longitude, school districts, distances to neighbors) were deliberately excluded so that all locational value concentrates in the single `community_area` feature; mostly-missing and low-signal columns were also dropped.

**3. Baseline model.** A ridge regression (hedonic) model was built first. Its per-neighborhood coefficients give an interpretable baseline map of neighborhood premiums to compare the final model against. However, it notably fails to identify Hyde park as an affluent neighbourhood of Chicago. 

**4. Final model and tuning.** The final model is an XGBoost regressor with native categorical support. Hyperparameters were tuned with Optuna (100 trials of TPE search over depth, learning rate, regularization, and sampling parameters) using 5-fold cross-validation on an 85% training split, with a held-out 15% test set. Test-set performance: R² = 0.8956, MAE ≈ \$85.4k, RMSE ≈ \$138k against a mean sale price of ≈ \$422k. Feature importance confirms `community_area` is the dominant predictor, validating the model's use for studying neighborhood effects. 

**5. Extracting the premium.** For each community area, the model's partial dependence — the average predicted price if every home were placed in that area — is computed and centered on the citywide mean. The resulting dollar deviations are mapped as a choropleth of Chicago. Unlike the ridge baseline, XGBoost correctly identifies Hyde Park as a high-premium area.

**6. Error analysis.** Out-of-fold residual diagnostics (5-fold CV: R² = 0.8820, median absolute relative error 19.3%) show the model systematically overpredicts the cheapest decile of sales — consistent with distress sales in the data — and identify high-noise neighborhoods worth future attention.

**7. Business scenario.** A hypothetical comparison of two 2024-buy/2025-sell investors on the North Side: one buys in Lincoln Park (the most expensive area), the other uses the model's premium trend from 2021-2023 data to pick North Center (fastest-rising premium). The model-guided investor comes out ~\$46k ahead.

**8. Deployment artifact.** The final model is packaged as a serialized pipeline (`artifacts/final_tree_pipeline.joblib`), so raw data can be directly converted to the model. The `artifacts/` folder contains: the model file, the notebook used to generate it, and the `preprocessing.py` module required to load it.

## Key takeaways

- Neighborhood is the single strongest driver of Chicago single-family home prices in this dataset.
- Premiums range from roughly +\$415k (Lincoln Park) to −\$218k (Riverdale) relative to the citywide mean.
- A gradient-boosted model captures neighborhood effects (e.g., Hyde Park) that a linear hedonic model misses.

## Potential improvements

- Use census data to add demographic information.
- Include other property classes: Condos, multi-family houses, etc.
- Investigate model failures in low-price decide, and high noise neighbourhoods.