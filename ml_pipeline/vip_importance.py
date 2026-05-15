"""R vip-based permutation importance utilities."""

import os
from typing import Dict, List

import numpy as np
import pandas as pd


os.environ.setdefault("RPY2_CFFI_MODE", "ABI")


def compute_vip_r2_importance(
    model,
    X: pd.DataFrame,
    y: pd.Series,
    feature_names: List[str] | None = None,
) -> Dict[str, float]:
    """Compute validation-fold permutation importance using R vip and an R² metric."""
    from rpy2 import robjects
    from rpy2.robjects import default_converter, numpy2ri, pandas2ri
    from rpy2.robjects.conversion import localconverter
    from rpy2.robjects.packages import importr
    from rpy2.rinterface import rternalize

    importr("vip")

    predictor_names = list(X.columns)
    importance_feature_names = feature_names or predictor_names
    X_filled = X.fillna(0.0)
    y_series = pd.Series(y, name="target")

    @rternalize
    def py_pred(newdata):
        with localconverter(default_converter + pandas2ri.converter):
            converted = robjects.conversion.rpy2py(newdata)
        if not isinstance(converted, pd.DataFrame):
            converted = pd.DataFrame(converted)
        predictions = np.asarray(
            model.predict(converted.loc[:, predictor_names].to_numpy()),
            dtype=float,
        )
        with localconverter(default_converter + numpy2ri.converter):
            return robjects.conversion.py2rpy(predictions)

    with localconverter(default_converter + pandas2ri.converter):
        r_train = robjects.conversion.py2rpy(X_filled)
        r_target = robjects.conversion.py2rpy(y_series)
    r_feature_names = robjects.StrVector(list(importance_feature_names))

    vi_permute = robjects.r(
        """
        function(train_df, target_vec, feature_names, pred_wrapper) {
          metric_rsq <- function(truth, estimate) {
            sst <- sum((truth - mean(truth))^2)
            if (sst == 0) {
              return(NA_real_)
            }
            1 - sum((truth - estimate)^2) / sst
          }

          wrapped_pred <- function(object, newdata) {
            pred_wrapper(newdata)
          }

          vip::vi_permute(
            object = NULL,
            feature_names = feature_names,
            train = train_df,
            target = target_vec,
            metric = metric_rsq,
            smaller_is_better = FALSE,
            pred_wrapper = wrapped_pred,
            nsim = 1,
            keep = FALSE
          )
        }
        """
    )

    result = vi_permute(r_train, r_target, r_feature_names, py_pred)

    with localconverter(default_converter + pandas2ri.converter):
        result_df = robjects.conversion.rpy2py(result)

    return {
        row["Variable"]: float(row["Importance"])
        for _, row in result_df.iterrows()
    }
