"""Preprocessing for the final Chicago neighborhood-premium model.

Any notebook or script that loads the
pipeline must be able to import this module first:

    from preprocessing import ChicagoPreprocessor
    pipeline = joblib.load("final_tree_pipeline.joblib")
"""

import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TARGET = "sale_price"

# Same exclusions as notebooks/Final_results2_tree.ipynb: features encoding
# relative location (location's contribution should be concentrated in the
# community_area feature), plus columns with little predictive value or
# mostly missing. See the Exploring Alternate Linear Regression Models
# notebook in the Data folder for details.
FEATURES_TO_EXCLUDE = [
    "pin",
    "is_multisale",
    "flood_fema_sfha",
    "sale_filter_same_sale_within_365",
    "sale_filter_less_than_10k",
    "single_v_multi_family",
    "zip_code",
    "longitude",
    "latitude",
    "sale_date_parsed",
    "class",
    "centroid_x_crs_3435",
    "centroid_y_crs_3435",
    "pin10",
    "nearest_metra_route_dist_ft",
    "nearest_new_construction_pin10",
    "school_elementary_district_name",
    "nearest_golf_course_dist_ft",
    "school_secondary_district_name",
    "township_code",
    "neighborhood_code",
    "sale_filter_deed_type",
    "nearest_cta_route_dist_ft",
    "nearest_vacant_land_pin10",
    "airport_noise_dnl",
    "attic_finish",
    "census_acs5_tract_geoid",
    "nearest_neighbor_1_pin10",
    "nearest_neighbor_2_pin10",
    "nearest_neighbor_3_pin10",
    "sale_date",
    "sale_year",
]

CAT_COLS = [
    "type_of_residence",
    "construction_quality",
    "garage_size",
    "basement_type",
    "ext_wall_material",
    "repair_condition",
    "basement_finish",
    "central_heating",
    "central_air",
    "community_area",
]


class ChicagoPreprocessor(BaseEstimator, TransformerMixin):
    """Drop unused columns and cast categorical columns to category dtypes
    whose levels are frozen at fit time.
    """

    def __init__(self, drop_cols, cat_cols):
        self.drop_cols = drop_cols
        self.cat_cols = cat_cols

    def fit(self, X, y=None):
        X = X.drop(columns=self.drop_cols, errors="ignore")
        self.cat_dtypes_ = {
            col: pd.CategoricalDtype(X[col].astype("category").cat.categories)
            for col in self.cat_cols
        }
        # Pin column order too: XGBoost is position-sensitive.
        self.feature_names_ = list(X.columns)
        return self

    def transform(self, X):
        X = X.drop(columns=self.drop_cols, errors="ignore").copy()
        for col, dtype in self.cat_dtypes_.items():
            X[col] = X[col].astype(dtype)
        return X[self.feature_names_]
