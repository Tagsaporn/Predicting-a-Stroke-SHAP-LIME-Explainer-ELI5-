"""
stats_analysis.py
=================
การวิเคราะห์ทางสถิติเพื่อตอบ **Q1**: ปัจจัยใดสัมพันธ์กับ Stroke อย่างมีนัยสำคัญ

การทดสอบที่ใช้ (เลือกตามชนิดของตัวแปร):
    * Categorical vs Binary target  -> Chi-square test of independence + Cramér's V
    * Continuous   vs Binary target -> Mann-Whitney U (ไม่ต้องสมมติ normality)
                                       + rank-biserial correlation (effect size)
    * ตัวแปรไบนารี vs target        -> Odds Ratio พร้อม 95% CI
    * ตรวจ normality ด้วย Shapiro-Wilk เพื่อ "พิสูจน์" ว่าเลือก Mann-Whitney ถูกต้อง

ปรับ p-value ด้วย Benjamini-Hochberg (FDR) เพราะทดสอบพร้อมกันหลายตัวแปร
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import stats

from . import config as cfg


# ---------------------------------------------------------------------------
# Effect sizes & helpers
# ---------------------------------------------------------------------------
def cramers_v(contingency: np.ndarray) -> float:
    """Cramér's V — ขนาดความสัมพันธ์ของตัวแปรเชิงกลุ่ม (0 ถึง 1)"""
    chi2 = stats.chi2_contingency(contingency, correction=False)[0]
    n = contingency.sum()
    if n == 0:
        return np.nan
    phi2 = chi2 / n
    r, k = contingency.shape
    # Bias correction (Bergsma, 2013)
    phi2corr = max(0.0, phi2 - (k - 1) * (r - 1) / (n - 1))
    rcorr = r - (r - 1) ** 2 / (n - 1)
    kcorr = k - (k - 1) ** 2 / (n - 1)
    denom = min(kcorr - 1, rcorr - 1)
    return float(np.sqrt(phi2corr / denom)) if denom > 0 else np.nan


def rank_biserial_from_u(u_stat: float, n1: int, n2: int) -> float:
    """
    Rank-biserial correlation จากค่า U ของ Mann-Whitney
    เครื่องหมายบวก = กลุ่ม 1 มีค่าสูงกว่ากลุ่ม 0
    """
    return float(2 * u_stat / (n1 * n2) - 1)


def cliffs_delta_from_u(u_stat: float, n1: int, n2: int) -> float:
    """Cliff's delta — ขนาดผลต่างสะสมระหว่าง 2 กลุ่ม (เทียบเท่า rank-biserial)"""
    return round(2 * u_stat / (n1 * n2) - 1, 4)


def benjamini_hochberg(p_values: pd.Series) -> pd.Series:
    """ปรับ p-value ด้วยวิธี Benjamini-Hochberg (FDR)"""
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return pd.Series(out, index=p_values.index, name="p_adjusted_bh")


def interpret_cramers_v(v: float) -> str:
    """ตีความขนาด effect size ของ Cramér's V (เกณฑ์ของ Cohen)"""
    if pd.isna(v):
        return "n/a"
    if v < 0.10:
        return "น้อยมาก (negligible)"
    if v < 0.20:
        return "น้อย (small)"
    if v < 0.30:
        return "ปานกลาง (medium)"
    return "มาก (large)"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------
def chi_square_vs_target(
    df: pd.DataFrame,
    cat_cols: list[str] | tuple[str, ...] | None = None,
    target: str = cfg.TARGET,
) -> pd.DataFrame:
    """ทดสอบ Chi-square ระหว่างตัวแปรเชิงกลุ่ม/ไบนารีกับ target"""
    if cat_cols is None:
        cat_cols = list(cfg.CATEGORICAL_FEATURES) + list(cfg.BINARY_FEATURES)

    rows = []
    for col in cat_cols:
        table = pd.crosstab(df[col], df[target])
        if table.shape[0] < 2 or table.shape[1] < 2:
            continue

        chi2, p, dof, expected = stats.chi2_contingency(table)
        n = int(table.to_numpy().sum())
        min_expected = float(expected.min())
        pct_expected_lt5 = float((expected < 5).mean() * 100)

        # สัดส่วนผู้ป่วย stroke ในแต่ละกลุ่ม
        rates = (table[1] / table.sum(axis=1) * 100).round(2)

        rows.append(
            {
                "feature": col,
                "test": "Chi-square",
                "chi2": round(float(chi2), 4),
                "dof": int(dof),
                "p_value": float(p),
                "cramers_v": round(cramers_v(table.to_numpy()), 4),
                "effect_size": interpret_cramers_v(cramers_v(table.to_numpy())),
                "n": n,
                "min_expected_count": round(min_expected, 2),
                "pct_cells_expected_lt5": round(pct_expected_lt5, 2),
                "warning": (
                    "มีการคาดหวัง < 5 เกิน 20% ของเซลล์ — ควรตีความอย่างระวัง"
                    if pct_expected_lt5 > 20
                    else ""
                ),
                "stroke_rate_by_group_%": " | ".join(
                    f"{idx}: {val}" for idx, val in rates.items()
                ),
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_adjusted_bh"] = benjamini_hochberg(result["p_value"])
        result["significant_0.05"] = result["p_adjusted_bh"] < 0.05
    return result


def mann_whitney_vs_target(
    df: pd.DataFrame,
    num_cols: list[str] | tuple[str, ...] | None = None,
    target: str = cfg.TARGET,
) -> pd.DataFrame:
    """ทดสอบ Mann-Whitney U ระหว่างตัวแปรต่อเนื่องกับ target (2 กลุ่ม)"""
    if num_cols is None:
        num_cols = list(cfg.NUMERIC_FEATURES)

    rows = []
    for col in num_cols:
        sub = df[[col, target]].dropna()
        g0 = sub.loc[sub[target] == 0, col]
        g1 = sub.loc[sub[target] == 1, col]
        if len(g0) < 3 or len(g1) < 3:
            continue

        u_stat, p = stats.mannwhitneyu(g1, g0, alternative="two-sided")
        rb = rank_biserial_from_u(u_stat, len(g1), len(g0))

        # ตรวจ normality ของแต่ละกลุ่ม (ถ้า p < .05 = ไม่ปกติ -> ยืนยันใช้ Mann-Whitney)
        _, p_norm0 = stats.shapiro(g0.sample(min(len(g0), 5000), random_state=cfg.RANDOM_STATE))
        _, p_norm1 = stats.shapiro(g1.sample(min(len(g1), 5000), random_state=cfg.RANDOM_STATE))

        rows.append(
            {
                "feature": col,
                "test": "Mann-Whitney U",
                "u_statistic": float(u_stat),
                "p_value": float(p),
                "rank_biserial": round(rb, 4),
                "median_no_stroke": round(float(g0.median()), 2),
                "median_stroke": round(float(g1.median()), 2),
                "mean_no_stroke": round(float(g0.mean()), 2),
                "mean_stroke": round(float(g1.mean()), 2),
                "shapiro_p_no_stroke": round(float(p_norm0), 6),
                "shapiro_p_stroke": round(float(p_norm1), 6),
                "normality_rejected": bool(p_norm0 < 0.05 and p_norm1 < 0.05),
            }
        )

    result = pd.DataFrame(rows)
    if not result.empty:
        result["p_adjusted_bh"] = benjamini_hochberg(result["p_value"])
        result["significant_0.05"] = result["p_adjusted_bh"] < 0.05
        result["direction"] = np.where(
            result["rank_biserial"] > 0,
            "กลุ่ม stroke มีค่าสูงกว่า",
            "กลุ่ม stroke มีค่าต่ำกว่า",
        )
    return result


def odds_ratios(
    df: pd.DataFrame,
    binary_cols: list[str] | tuple[str, ...] | None = None,
    target: str = cfg.TARGET,
) -> pd.DataFrame:
    """Odds Ratio + 95% CI สำหรับตัวแปรไบนารี (hypertension, heart_disease)"""
    if binary_cols is None:
        binary_cols = list(cfg.BINARY_FEATURES)

    rows = []
    for col in binary_cols:
        table = pd.crosstab(df[col], df[target]).reindex(
            index=[0, 1], columns=[0, 1], fill_value=0
        )
        a = table.loc[1, 1]  # exposed & stroke
        b = table.loc[1, 0]  # exposed & no stroke
        c = table.loc[0, 1]  # unexposed & stroke
        d = table.loc[0, 0]  # unexposed & no stroke

        # Haldane-Anscombe correction ถ้ามีเซลล์เป็น 0
        if min(a, b, c, d) == 0:
            a, b, c, d = a + 0.5, b + 0.5, c + 0.5, d + 0.5

        or_val = (a * d) / (b * c)
        se = np.sqrt(1 / a + 1 / b + 1 / c + 1 / d)
        ci_low, ci_high = np.exp(np.log(or_val) - 1.96 * se), np.exp(np.log(or_val) + 1.96 * se)

        rows.append(
            {
                "feature": col,
                "odds_ratio": round(float(or_val), 3),
                "ci_95_low": round(float(ci_low), 3),
                "ci_95_high": round(float(ci_high), 3),
                "stroke_rate_exposed_%": round(100 * a / (a + b), 2),
                "stroke_rate_unexposed_%": round(100 * c / (c + d), 2),
                "interpretation": "เพิ่มความเสี่ยง" if or_val > 1 else "ลดความเสี่ยง",
            }
        )
    return pd.DataFrame(rows)


def describe_by_target(
    df: pd.DataFrame,
    num_cols: list[str] | tuple[str, ...] | None = None,
    target: str = cfg.TARGET,
) -> pd.DataFrame:
    """สถิติพรรณนาของตัวแปรต่อเนื่อง แยกตามกลุ่ม target"""
    if num_cols is None:
        num_cols = list(cfg.NUMERIC_FEATURES)

    rows = []
    for col in num_cols:
        for grp in (0, 1):
            s = df.loc[df[target] == grp, col].dropna()
            rows.append(
                {
                    "feature": col,
                    "group": "stroke" if grp == 1 else "no_stroke",
                    "n": int(len(s)),
                    "mean": round(float(s.mean()), 2),
                    "std": round(float(s.std()), 2),
                    "median": round(float(s.median()), 2),
                    "q1": round(float(s.quantile(0.25)), 2),
                    "q3": round(float(s.quantile(0.75)), 2),
                    "min": round(float(s.min()), 2),
                    "max": round(float(s.max()), 2),
                }
            )
    return pd.DataFrame(rows)


def correlation_with_target(
    df: pd.DataFrame,
    num_cols: list[str] | tuple[str, ...] | None = None,
    target: str = cfg.TARGET,
) -> pd.DataFrame:
    """
    Spearman correlation ระหว่างตัวแปรต่อเนื่อง/ไบนารีกับ target
    (ใช้ Spearman เพราะไม่ต้องสมมติ linearity และทนต่อ outlier)
    """
    if num_cols is None:
        num_cols = list(cfg.NUMERIC_FEATURES) + list(cfg.BINARY_FEATURES)

    rows = []
    for col in num_cols:
        sub = df[[col, target]].dropna()
        rho, p = stats.spearmanr(sub[col], sub[target])
        rows.append(
            {
                "feature": col,
                "spearman_rho": round(float(rho), 4),
                "p_value": float(p),
            }
        )
    result = pd.DataFrame(rows)
    result["p_adjusted_bh"] = benjamini_hochberg(result["p_value"])
    return result.sort_values("spearman_rho", key=abs, ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
@dataclass
class Q1Results:
    """รวมผลลัพธ์ทั้งหมดของ Q1 ไว้ที่เดียว"""

    chi_square: pd.DataFrame
    mann_whitney: pd.DataFrame
    odds_ratios: pd.DataFrame
    descriptives: pd.DataFrame
    correlations: pd.DataFrame

    def to_dict(self) -> dict[str, pd.DataFrame]:
        return {
            "q1_chi_square": self.chi_square,
            "q1_mann_whitney": self.mann_whitney,
            "q1_odds_ratios": self.odds_ratios,
            "q1_descriptives": self.descriptives,
            "q1_correlations": self.correlations,
        }


def run_q1_analysis(df: pd.DataFrame, target: str = cfg.TARGET) -> Q1Results:
    """รันการทดสอบทางสถิติทั้งหมดของ Q1"""
    return Q1Results(
        chi_square=chi_square_vs_target(df, target=target),
        mann_whitney=mann_whitney_vs_target(df, target=target),
        odds_ratios=odds_ratios(df, target=target),
        descriptives=describe_by_target(df, target=target),
        correlations=correlation_with_target(df, target=target),
    )
