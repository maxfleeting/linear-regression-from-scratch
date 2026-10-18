from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from linear_regression import LinearRegressionScratch, manual_metrics


def test_exact_linear_relation() -> None:
    x = np.array([0.0, 1.0, 2.0, 3.0, 4.0])
    y = 3.0 + 2.5 * x
    model = LinearRegressionScratch().fit(x, y)

    assert np.isclose(model.intercept_, 3.0)
    assert np.allclose(model.coef_, [2.5])
    assert np.allclose(model.predict([5.0, 6.0]), [15.5, 18.0])


def test_scratch_matches_sklearn_on_assignment_data() -> None:
    data_path = Path(__file__).parents[1] / "data" / "study_hours_scores.csv"
    data = pd.read_csv(data_path)
    x_train, x_test, y_train, _ = train_test_split(
        data[["study_hours"]],
        data["score"],
        test_size=0.25,
        random_state=42,
    )

    scratch = LinearRegressionScratch().fit(x_train, y_train)
    sklearn_model = LinearRegression().fit(x_train, y_train)

    assert np.allclose(scratch.predict(x_test), sklearn_model.predict(x_test), atol=1e-10)
    assert np.isclose(scratch.intercept_, sklearn_model.intercept_, atol=1e-10)
    assert np.allclose(scratch.coef_, sklearn_model.coef_, atol=1e-10)


def test_manual_metrics_for_perfect_prediction() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    result = manual_metrics(values, values)

    assert result.mse == 0.0
    assert result.rmse == 0.0
    assert result.mae == 0.0
    assert result.r2 == 1.0
