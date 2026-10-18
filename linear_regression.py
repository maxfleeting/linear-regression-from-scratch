"""第4章线性回归：从零实现最小二乘法，并与 scikit-learn 对照。

作者：朱慧恒
学号：2263372
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split


AUTHOR = "朱慧恒"
STUDENT_ID = "2263372"


@dataclass(frozen=True)
class RegressionMetrics:
    """保存一组回归评估指标。"""

    mse: float
    rmse: float
    mae: float
    r2: float

    def as_dict(self) -> dict[str, float]:
        return {
            "MSE": self.mse,
            "RMSE": self.rmse,
            "MAE": self.mae,
            "R2": self.r2,
        }


class LinearRegressionScratch:
    """用最小二乘法闭式解实现多元线性回归。

    目标是最小化平方误差和：||Xw - y||²。
    为了在特征相关或矩阵不可逆时仍能稳定求解，这里使用 Moore-Penrose
    伪逆：theta = pinv(X_design) @ y。
    """

    def __init__(self, fit_intercept: bool = True) -> None:
        self.fit_intercept = fit_intercept
        self.coef_: np.ndarray | None = None
        self.intercept_: float = 0.0

    @staticmethod
    def _as_2d_array(features: np.ndarray | pd.DataFrame | pd.Series) -> np.ndarray:
        array = np.asarray(features, dtype=float)
        if array.ndim == 1:
            array = array.reshape(-1, 1)
        if array.ndim != 2:
            raise ValueError("特征 X 必须是一维或二维数组。")
        return array

    def fit(
        self,
        features: np.ndarray | pd.DataFrame | pd.Series,
        targets: np.ndarray | pd.Series,
    ) -> "LinearRegressionScratch":
        x = self._as_2d_array(features)
        y = np.asarray(targets, dtype=float).reshape(-1)
        if x.shape[0] != y.shape[0]:
            raise ValueError("X 与 y 的样本数量必须一致。")
        if x.shape[0] == 0:
            raise ValueError("训练数据不能为空。")

        design = np.column_stack([np.ones(x.shape[0]), x]) if self.fit_intercept else x
        theta = np.linalg.pinv(design) @ y

        if self.fit_intercept:
            self.intercept_ = float(theta[0])
            self.coef_ = np.asarray(theta[1:], dtype=float)
        else:
            self.intercept_ = 0.0
            self.coef_ = np.asarray(theta, dtype=float)
        return self

    def predict(self, features: np.ndarray | pd.DataFrame | pd.Series) -> np.ndarray:
        if self.coef_ is None:
            raise RuntimeError("模型尚未训练，请先调用 fit()。")
        x = self._as_2d_array(features)
        if x.shape[1] != self.coef_.shape[0]:
            raise ValueError("预测数据的特征数量与训练数据不一致。")
        return self.intercept_ + x @ self.coef_


def manual_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """不依赖 sklearn，手写 MSE、RMSE、MAE 和 R²。"""

    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)
    residuals = actual - predicted
    mse = float(np.mean(residuals**2))
    mae = float(np.mean(np.abs(residuals)))
    total_sum_squares = float(np.sum((actual - np.mean(actual)) ** 2))
    residual_sum_squares = float(np.sum(residuals**2))
    r2 = 1.0 - residual_sum_squares / total_sum_squares
    return RegressionMetrics(mse=mse, rmse=float(np.sqrt(mse)), mae=mae, r2=r2)


def sklearn_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> RegressionMetrics:
    """使用 sklearn 计算同一组指标，用于交叉验证手写实现。"""

    mse = float(mean_squared_error(y_true, y_pred))
    return RegressionMetrics(
        mse=mse,
        rmse=float(np.sqrt(mse)),
        mae=float(mean_absolute_error(y_true, y_pred)),
        r2=float(r2_score(y_true, y_pred)),
    )


def configure_chinese_font() -> None:
    plt.rcParams["font.sans-serif"] = [
        "Microsoft YaHei",
        "SimHei",
        "Noto Sans CJK SC",
        "Arial Unicode MS",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def save_fit_plot(
    data: pd.DataFrame,
    model: LinearRegressionScratch,
    metrics: RegressionMetrics,
    output_path: Path,
    *,
    cover: bool = False,
) -> None:
    configure_chinese_font()
    fig, ax = plt.subplots(figsize=(16, 9) if cover else (12, 7.2), dpi=120)
    background = "#071426" if cover else "#F6F8FC"
    foreground = "#F8FAFC" if cover else "#10233F"
    fig.patch.set_facecolor(background)
    ax.set_facecolor(background)

    x_all = data[["study_hours"]].to_numpy()
    y_all = data["score"].to_numpy()
    x_line = np.linspace(x_all.min(), x_all.max(), 240).reshape(-1, 1)
    y_line = model.predict(x_line)

    ax.scatter(
        x_all,
        y_all,
        s=70 if cover else 55,
        color="#44D7B6",
        edgecolor="#071426" if cover else "white",
        linewidth=0.9,
        alpha=0.92,
        label="教学模拟样本",
        zorder=3,
    )
    ax.plot(
        x_line,
        y_line,
        color="#FFB84D",
        linewidth=4 if cover else 3,
        label="最小二乘拟合直线",
        zorder=4,
    )

    if cover:
        fig.text(
            0.052,
            0.945,
            "从零实现线性回归",
            color=foreground,
            fontsize=31,
            fontweight="bold",
            ha="left",
            va="top",
        )
        fig.text(
            0.052,
            0.875,
            "最小二乘法预测学习成绩",
            color=foreground,
            fontsize=22,
            fontweight="bold",
            ha="left",
            va="top",
        )
        fig.text(
            0.052,
            0.817,
            f"第4章 · 算法实现   |   {AUTHOR} · {STUDENT_ID}   |   我的印记",
            color="#9FB4CC",
            fontsize=14,
            ha="left",
            va="top",
        )
        ax.text(
            0.985,
            0.06,
            f"ŷ = {model.intercept_:.2f} + {model.coef_[0]:.2f}x\nR² = {metrics.r2:.3f}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=17,
            color=foreground,
            bbox={"boxstyle": "round,pad=0.7", "facecolor": "#102A43", "edgecolor": "#2F80ED"},
        )
    else:
        ax.set_title("线性回归拟合结果", fontsize=22, fontweight="bold", color=foreground, pad=14)
        ax.text(
            0.02,
            0.95,
            f"ŷ = {model.intercept_:.2f} + {model.coef_[0]:.2f}x\n"
            f"测试集 MSE = {metrics.mse:.3f}\n测试集 R² = {metrics.r2:.3f}",
            transform=ax.transAxes,
            va="top",
            fontsize=12,
            color=foreground,
            bbox={"boxstyle": "round,pad=0.6", "facecolor": "white", "edgecolor": "#C8D4E3"},
        )

    ax.set_xlabel("学习时长（小时/天）", fontsize=14, color=foreground, labelpad=10)
    ax.set_ylabel("考试成绩（分）", fontsize=14, color=foreground, labelpad=10)
    ax.grid(True, linestyle="--", alpha=0.18 if cover else 0.35)
    ax.tick_params(colors=foreground, labelsize=11)
    for spine in ax.spines.values():
        spine.set_visible(False)
    legend = ax.legend(loc="lower right" if not cover else "upper left", frameon=False, fontsize=12)
    for text in legend.get_texts():
        text.set_color(foreground)

    if cover:
        fig.subplots_adjust(left=0.052, right=0.992, bottom=0.09, top=0.75)
    else:
        fig.tight_layout(pad=2.0)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def save_residual_plot(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: Path,
) -> None:
    configure_chinese_font()
    residuals = np.asarray(y_true) - np.asarray(y_pred)
    fig, ax = plt.subplots(figsize=(11, 6.6), dpi=120)
    fig.patch.set_facecolor("#F6F8FC")
    ax.set_facecolor("#F6F8FC")
    ax.scatter(y_pred, residuals, s=65, color="#2F80ED", edgecolor="white", linewidth=1.0)
    ax.axhline(0, color="#E05263", linewidth=2, linestyle="--")
    ax.set_title("测试集残差图", fontsize=21, fontweight="bold", color="#10233F")
    ax.set_xlabel("预测成绩", fontsize=13)
    ax.set_ylabel("残差（真实值 - 预测值）", fontsize=13)
    ax.grid(True, linestyle="--", alpha=0.35)
    for spine in ax.spines.values():
        spine.set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, facecolor=fig.get_facecolor(), bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="从零实现最小二乘线性回归")
    parser.add_argument(
        "--data",
        type=Path,
        default=Path(__file__).parent / "data" / "study_hours_scores.csv",
        help="CSV 数据路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).parent / "outputs",
        help="结果输出目录",
    )
    parser.add_argument("--test-size", type=float, default=0.25, help="测试集比例")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(args.data)
    required_columns = {"study_hours", "score"}
    missing = required_columns - set(data.columns)
    if missing:
        raise ValueError(f"数据缺少必要列：{sorted(missing)}")

    features = data[["study_hours"]]
    targets = data["score"]
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        targets,
        test_size=args.test_size,
        random_state=args.seed,
    )

    scratch_model = LinearRegressionScratch().fit(x_train, y_train)
    scratch_predictions = scratch_model.predict(x_test)
    scratch_result = manual_metrics(y_test.to_numpy(), scratch_predictions)

    sklearn_model = LinearRegression().fit(x_train, y_train)
    sklearn_predictions = sklearn_model.predict(x_test)
    sklearn_result = sklearn_metrics(y_test.to_numpy(), sklearn_predictions)
    maximum_difference = float(np.max(np.abs(scratch_predictions - sklearn_predictions)))

    metrics_payload = {
        "author": AUTHOR,
        "student_id": STUDENT_ID,
        "dataset": "教学模拟数据：学习时长与考试成绩",
        "split": {
            "train_samples": int(len(x_train)),
            "test_samples": int(len(x_test)),
            "test_size": args.test_size,
            "random_seed": args.seed,
        },
        "scratch_model": {
            "intercept": scratch_model.intercept_,
            "coefficient": float(scratch_model.coef_[0]),
            **scratch_result.as_dict(),
        },
        "sklearn_model": {
            "intercept": float(sklearn_model.intercept_),
            "coefficient": float(sklearn_model.coef_[0]),
            **sklearn_result.as_dict(),
        },
        "max_prediction_difference": maximum_difference,
    }
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    predictions = pd.DataFrame(
        {
            "study_hours": x_test["study_hours"].to_numpy(),
            "actual_score": y_test.to_numpy(),
            "scratch_prediction": scratch_predictions,
            "sklearn_prediction": sklearn_predictions,
            "residual": y_test.to_numpy() - scratch_predictions,
        }
    ).sort_values("study_hours")
    predictions.to_csv(args.output_dir / "predictions.csv", index=False, encoding="utf-8-sig")

    save_fit_plot(data, scratch_model, scratch_result, args.output_dir / "fit_plot.png")
    save_fit_plot(
        data,
        scratch_model,
        scratch_result,
        args.output_dir / "submission_cover.png",
        cover=True,
    )
    save_residual_plot(
        y_test.to_numpy(),
        scratch_predictions,
        args.output_dir / "residual_plot.png",
    )

    print("=" * 64)
    print("第4章 线性回归｜算法实现")
    print(f"作者：{AUTHOR}    学号：{STUDENT_ID}")
    print("=" * 64)
    # 使用纯 ASCII 的 y_hat，避免部分 Windows GBK 终端无法编码数学字符 ŷ。
    print(f"手写模型：y_hat = {scratch_model.intercept_:.4f} + {scratch_model.coef_[0]:.4f}x")
    print(f"MSE={scratch_result.mse:.4f}  RMSE={scratch_result.rmse:.4f}  "
          f"MAE={scratch_result.mae:.4f}  R²={scratch_result.r2:.4f}")
    print(f"与 sklearn 预测结果的最大绝对差：{maximum_difference:.12f}")
    print(f"结果目录：{args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
