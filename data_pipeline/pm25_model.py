"""One-hour PM2.5 forecasting research baseline, restricted to UCI Beijing.

This is not a health, fitness-for-duty, or evacuation classifier. The reference
series is stationary ambient monitoring; this model must not be used on 10-second
SPS30 readings, personal exposure streams, or a moving fireground.

Reproduce: python pm25_model.py --train
Example:   python pm25_model.py --demo
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import pickle
import platform
import sys
import urllib.request
import zipfile
from datetime import datetime, timedelta, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


BASE = Path(__file__).resolve().parent
DOMAIN = "uci_beijing_reference"
VERSION = "pm25-hourly-0.1.0"
SOURCE_URL = "https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data"
DOWNLOAD_URL = "https://archive.ics.uci.edu/static/public/381/beijing%2Bpm2%2B5%2Bdata.zip"
SOURCE_SHA256 = "4127f868775e31b3956522adc0ec75af8937dde6a3896e8beed3a376c6d27f1c"
LOCAL_TZ = timezone(timedelta(hours=8))
FEATURE_NAMES = [
    "pm25_lag0", "pm25_lag1", "pm25_lag2", "pm25_lag3", "pm25_lag6", "pm25_lag12", "pm25_lag24",
    "pm25_mean3", "pm25_mean6", "pm25_mean12", "pm25_mean24",
    "pm25_std6", "pm25_std24", "pm25_min24", "pm25_max24",
    "pm25_slope3", "pm25_slope6", "pm25_slope24",
]


def _json_write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def _features(window):
    """25 consecutive observed hours t-24..t; never inspect a future value."""
    values = np.asarray(window, dtype=float)
    if len(values) != 25 or not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Feature window must contain 25 finite, nonnegative hourly observations")
    return np.asarray(
        [values[-1 - lag] for lag in (0, 1, 2, 3, 6, 12, 24)]
        + [values[-n:].mean() for n in (3, 6, 12, 24)]
        + [values[-n:].std(ddof=0) for n in (6, 24)]
        + [values[-24:].min(), values[-24:].max()]
        + [(values[-1] - values[-1 - n]) / n for n in (3, 6, 24)],
        dtype=float,
    )


def _download_source(destination):
    """Download the official archive and verify the exact reviewed CSV bytes."""
    destination = Path(destination)
    limit = 20 * 1024 * 1024
    with urllib.request.urlopen(DOWNLOAD_URL, timeout=60) as response:
        archive_bytes = response.read(limit + 1)
    if len(archive_bytes) > limit:
        raise ValueError("Official archive exceeded the expected download size")
    with zipfile.ZipFile(io.BytesIO(archive_bytes)) as archive:
        member = archive.getinfo("PRSA_data_2010.1.1-2014.12.31.csv")
        if member.file_size > limit:
            raise ValueError("Official CSV exceeded the expected extracted size")
        payload = archive.read(member)
    if hashlib.sha256(payload).hexdigest() != SOURCE_SHA256:
        raise ValueError("Downloaded source checksum differs from the reviewed UCI CSV")
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_bytes(payload)
    return destination


def _read_source(path=None):
    source = Path(path) if path else BASE / "data" / "pm25_source.csv"
    if not source.is_file():
        if path is not None:
            raise FileNotFoundError(f"Requested source does not exist: {source}")
        _download_source(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    if digest != SOURCE_SHA256:
        raise ValueError("Source checksum differs from the reviewed UCI archive; do not train silently on a replacement")
    frame = pd.read_csv(source, na_values=["NA"])
    times = pd.to_datetime(frame[["year", "month", "day", "hour"]])
    if len(frame) != 43824 or times.duplicated().any() or not times.is_monotonic_increasing:
        raise ValueError("Unexpected UCI source length, duplicates, or timestamp ordering")
    if not (times.diff().dropna() == pd.Timedelta(hours=1)).all():
        raise ValueError("Source is not on the expected exact hourly grid")
    frame = frame.assign(timestamp=times.dt.tz_localize(LOCAL_TZ), pm25_ug_m3=pd.to_numeric(frame["pm2.5"], errors="coerce"))
    return frame


def _supervised(frame):
    values = frame["pm25_ug_m3"].to_numpy(dtype=float)
    all_features, targets, target_indices = [], [], []
    for t in range(24, len(values) - 1):
        # Require the entire past window AND the next-hour target to be observed.
        # No interpolation, forward filling, or compression across missing hours.
        full_window = values[t - 24:t + 2]
        if not np.all(np.isfinite(full_window)) or np.any(full_window < 0):
            continue
        all_features.append(_features(values[t - 24:t + 1]))
        targets.append(values[t + 1])
        target_indices.append(t + 1)
    indices = np.asarray(target_indices, dtype=int)
    return np.asarray(all_features), np.asarray(targets), indices, frame.iloc[indices]["timestamp"].reset_index(drop=True)


def _metrics(y, predicted):
    residual = np.asarray(y) - np.asarray(predicted)
    return {
        "n": int(len(y)),
        "mae_ug_m3": float(np.mean(np.abs(residual))),
        "rmse_ug_m3": float(np.sqrt(np.mean(residual ** 2))),
        "p90_absolute_error_ug_m3": float(np.quantile(np.abs(residual), 0.9)),
    }


def _predict(estimator, x):
    return np.maximum(0.0, x[:, 0] if estimator is None else estimator.predict(x))


def _calibration_radius(residuals):
    errors = np.sort(np.abs(np.asarray(residuals, dtype=float)))
    rank = min(len(errors), math.ceil((len(errors) + 1) * 0.90))
    return float(errors[rank - 1]), int(rank)


def _coverage(y, predicted, radius):
    lower, upper = np.maximum(0, predicted - radius), predicted + radius
    return float(np.mean((y >= lower) & (y <= upper)))


def _split_summary(times, mask, raw_start, raw_end):
    subset = times[mask]
    return {
        "target_start_inclusive": raw_start,
        "target_end_exclusive": raw_end,
        "usable_samples": int(mask.sum()),
        "first_usable_target": subset.iloc[0].isoformat(),
        "last_usable_target": subset.iloc[-1].isoformat(),
    }


def _make_report(metrics, predictions):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Fixed display period selected by calendar, not forecast performance.
    shown = predictions[(predictions["target_timestamp"] >= "2014-01-01") & (predictions["target_timestamp"] < "2014-01-08")].copy()
    x = pd.to_datetime(shown["target_timestamp"], utc=True)
    figure, ax = plt.subplots(figsize=(12, 4.4), constrained_layout=True)
    ax.fill_between(x, shown["interval_lower_ug_m3"].to_numpy(), shown["interval_upper_ug_m3"].to_numpy(), color="#6c8fff", alpha=0.18, label="Calibration-residual interval (nominal 90%)")
    ax.plot(x, shown["actual_pm25_ug_m3"], color="#142433", linewidth=1.7, label="Observed next hour")
    ax.plot(x, shown["predicted_pm25_ug_m3"], color="#4164d6", linewidth=1.2, label=metrics["selected_model"])
    ax.plot(x, shown["persistence_pm25_ug_m3"], color="#a46d2f", linewidth=0.85, alpha=0.75, label="Persistence baseline")
    ax.set_ylabel("PM2.5 (ug/m3)")
    ax.set_xlabel("Forecast target time (UTC); source calendar interpreted as UTC+08:00")
    ax.set_title("Held-out hourly Beijing example: 2014-01-01 to 2014-01-07")
    ax.grid(alpha=0.2)
    ax.legend(loc="upper right", fontsize=8)
    figure.savefig(BASE / "reports" / "pm25_prediction.png", dpi=150)
    plt.close(figure)
    test = metrics["test"]["selected"]
    baseline = metrics["test"]["persistence"]
    candidate_rows = "".join(f"<tr><td>{name}</td><td>{item['validation']['mae_ug_m3']:.3f}</td><td>{item['validation']['rmse_ug_m3']:.3f}</td></tr>" for name, item in metrics["candidates"].items())
    split_rows = "".join(f"<tr><td>{name}</td><td>{s['target_start_inclusive']} 至 {s['target_end_exclusive']}（不含）</td><td>{s['usable_samples']:,}</td></tr>" for name, s in metrics["splits"].items())
    comparison_note = (
        "独立测试的 MAE 没有超过保持上一小时值的简单基线；验证集上的小幅改进未保持到测试期，因此不能宣称预测精度得到提升。"
        if test["mae_ug_m3"] >= baseline["mae_ug_m3"]
        else "独立测试的 MAE 低于简单基线；这一历史结果仍不能证明能够迁移到新设备、新地点或火场。"
    )
    html = f"""<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>PM2.5 小时预测研究</title><style>body{{font:16px/1.7 system-ui,'Microsoft YaHei',sans-serif;color:#193149;background:#f4f7fb;margin:0;padding:32px}}main{{max-width:1050px;margin:auto;background:white;padding:32px;border-radius:18px}}h1{{font-size:28px;margin-top:0}}h2{{font-size:21px;margin-top:30px}}p{{max-width:95ch}}.notice{{background:#fff3dd;padding:16px;border-left:5px solid #cb892e}}.cards{{display:flex;gap:16px;flex-wrap:wrap}}.card{{background:#edf3fc;padding:18px;border-radius:12px;min-width:180px}}.card strong{{font-size:28px;display:block}}table{{border-collapse:collapse;width:100%;font-size:14px}}td,th{{padding:10px;border-bottom:1px solid #d9e3ee;text-align:left}}img{{width:100%;height:auto}}code{{background:#eaf0f6;padding:2px 5px}}a{{color:#255bc1}}</style><main><h1>PM2.5 下一小时浓度预测</h1><p class="notice">这是固定北京监测点的历史研究模型，不是健康、适岗或撤离分类器。禁止直接用于 SPS30 的 10 秒数据、个人暴露或移动火场。接口仅在明确指定 <code>uci_beijing_reference</code> 且数据满足小时契约时返回预测。</p><p>使用真实 UCI 2010—2014 年逐小时记录；只使用历史 PM2.5 滞后、滚动统计和变化率。未使用天气、CO、人员身份或未来读数。验证集 MAE 最优模型为 <b>{metrics['selected_model']}</b>。</p><div class="cards"><div class="card">2014 测试 MAE<strong>{test['mae_ug_m3']:.2f}</strong>µg/m³</div><div class="card">保持上一小时值 MAE<strong>{baseline['mae_ug_m3']:.2f}</strong>µg/m³</div><div class="card">测试区间实际覆盖率<strong>{100*metrics['test']['interval_coverage']:.1f}%</strong>名义 90% 残差区间</div></div><h2>独立测试示例</h2><img src="pm25_prediction.png" alt="固定选择的2014年1月第一周逐小时真实值、预测值、基线和校准残差区间"><p>每一小时使用当时已观察到的历史，做一次下一小时预测；不是连续多步外推。图示日期固定按日历选择，全部可用测试记录均纳入指标。</p><h2>按时间分割</h2><table><tr><th>用途</th><th>目标时间范围（源日期解释为 UTC+08:00）</th><th>有效样本</th></tr>{split_rows}</table><p>每条输入使用 t−24 至 t 的 25 个连续观测，预测 t+1。缺测窗口和缺测目标直接剔除。分割根据目标时刻；边界处可使用此前已经观测到的历史，不使用边界之后的目标训练。</p><h2>验证集模型选择</h2><table><tr><th>候选</th><th>验证 MAE（µg/m³）</th><th>验证 RMSE（µg/m³）</th></tr>{candidate_rows}</table><p>全部候选只用训练集拟合，按验证 MAE 选择。选中后不重新拟合；2013 年下半年仅用于残差区间，2014 年只用于最后测试。</p><h2>区间如何解读</h2><p>在独立校准期计算绝对预测误差的经验分位数，得到固定半宽 {metrics['calibration']['absolute_residual_radius_ug_m3']:.3f} µg/m³，区间下限裁为零。时间序列可能发生分布变化，不能保证未来覆盖率；它不是健康概率，也没有撤离阈值。</p><h2>数据与复现</h2><p><a href="{SOURCE_URL}">UCI Beijing PM2.5 官方数据页</a>，DOI <a href="https://doi.org/10.24432/C5JS49">10.24432/C5JS49</a>，CC BY 4.0。原始资料未在页面明示时区；本包把源日历时间解释为 UTC+08:00，并在清单记录该假设。</p><p>在项目目录运行 <code>python pm25_model.py --train</code> 重现；运行 <code>python pm25_model.py --demo</code> 查看一个独立测试示例。完整指标见 <a href="pm25_metrics.json">pm25_metrics.json</a>，模型和特征清单见 <a href="../models/pm25_manifest.json">pm25_manifest.json</a>。</p></main></html>"""
    html = html.replace("<h2>独立测试示例</h2>", f"<p class=\"notice\"><b>比较结论：</b>{comparison_note}</p><h2>独立测试示例</h2>")
    (BASE / "reports" / "pm25_report.html").write_text(html, encoding="utf-8")


def train():
    for name in ("models", "reports", "data", "sources"):
        (BASE / name).mkdir(parents=True, exist_ok=True)
    frame = _read_source()
    x, y, indices, times = _supervised(frame)
    limits = {
        "train": ("2010-01-01", "2013-01-01"),
        "validation": ("2013-01-01", "2013-07-01"),
        "calibration": ("2013-07-01", "2014-01-01"),
        "test": ("2014-01-01", "2015-01-01"),
    }
    masks = {name: ((times >= pd.Timestamp(start, tz=LOCAL_TZ)) & (times < pd.Timestamp(end, tz=LOCAL_TZ))).to_numpy() for name, (start, end) in limits.items()}
    if any(not mask.any() for mask in masks.values()):
        raise ValueError("Empty chronological partition")
    candidates = {
        "persistence": None,
        "ridge_alpha_1": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "ridge_alpha_100": make_pipeline(StandardScaler(), Ridge(alpha=100.0)),
        "histgb_depth3": HistGradientBoostingRegressor(max_iter=140, learning_rate=0.06, max_depth=3, l2_regularization=10.0, early_stopping=False, random_state=42),
        "histgb_leaf15": HistGradientBoostingRegressor(max_iter=160, learning_rate=0.05, max_leaf_nodes=15, min_samples_leaf=40, l2_regularization=20.0, early_stopping=False, random_state=42),
    }
    candidate_results = {}
    for name, estimator in candidates.items():
        if estimator is not None:
            estimator.fit(x[masks["train"]], y[masks["train"]])
        prediction = _predict(estimator, x[masks["validation"]])
        candidate_results[name] = {"validation": _metrics(y[masks["validation"]], prediction)}
        print(f"{name}: validation MAE={candidate_results[name]['validation']['mae_ug_m3']:.4f}", flush=True)
    selected_name = min(candidate_results, key=lambda name: candidate_results[name]["validation"]["mae_ug_m3"])
    selected = candidates[selected_name]
    calibration_predictions = _predict(selected, x[masks["calibration"]])
    radius, quantile_rank = _calibration_radius(y[masks["calibration"]] - calibration_predictions)
    test_predictions = _predict(selected, x[masks["test"]])
    persistence = _predict(None, x[masks["test"]])
    selected_metrics, baseline_metrics = _metrics(y[masks["test"]], test_predictions), _metrics(y[masks["test"]], persistence)
    selected_metrics["mae_relative_improvement_vs_persistence"] = float(1 - selected_metrics["mae_ug_m3"] / baseline_metrics["mae_ug_m3"])
    metrics = {
        "model_version": VERSION,
        "task": "one-hour-ahead ambient PM2.5 regression; not health or evacuation classification",
        "source_domain": DOMAIN,
        "units": "ug/m3",
        "raw_rows": len(frame),
        "raw_pm25_missing_rows": int(frame["pm25_ug_m3"].isna().sum()),
        "supervised_usable_rows": len(y),
        "excluded_candidate_windows": int(len(frame) - 25 - len(y)),
        "window_policy": "25 consecutive observations t-24..t and observed target t+1; drop any missing/invalid window; no imputation",
        "split_policy": "target timestamp; no random split; historical context may cross a partition boundary",
        "selection_metric": "minimum validation MAE; selected estimator remains fit only on training data",
        "selected_model": selected_name,
        "splits": {name: _split_summary(times, masks[name], *limits[name]) for name in limits},
        "candidates": candidate_results,
        "calibration": {
            **_metrics(y[masks["calibration"]], calibration_predictions),
            "absolute_residual_radius_ug_m3": radius,
            "order_statistic_rank_one_based": quantile_rank,
            "nominal_coverage": 0.90,
            "in_sample_calibration_coverage": _coverage(y[masks["calibration"]], calibration_predictions, radius),
            "method": "ceil((n+1)*0.90)-th absolute residual; symmetric additive interval, lower bound clipped at 0",
            "limitation": "Empirical temporal calibration; no guaranteed future coverage under dependent data or domain shift; not a health probability",
        },
        "test": {"selected": selected_metrics, "persistence": baseline_metrics, "interval_coverage": _coverage(y[masks["test"]], test_predictions, radius)},
    }
    artifact = {
        "version": VERSION, "source_domain": DOMAIN, "cadence_seconds": 3600,
        "history_observations": 25, "feature_names": FEATURE_NAMES,
        "selected_model": selected_name, "estimator": selected,
        "interval_radius_ug_m3": radius,
    }
    model_path = BASE / "models" / "pm25_model.joblib"
    joblib.dump(artifact, model_path, compress=3)
    manifest = {
        **{key: value for key, value in artifact.items() if key != "estimator"},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_page": SOURCE_URL, "source_download": DOWNLOAD_URL,
        "citation": "Chen, S. (2015). Beijing PM2.5 [Dataset]. UCI Machine Learning Repository. https://doi.org/10.24432/C5JS49",
        "license": "CC BY 4.0", "license_url": "https://creativecommons.org/licenses/by/4.0/",
        "source_csv_sha256": SOURCE_SHA256,
        "model_file": "pm25_model.joblib", "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
        "timezone_assumption": "Source calendar interpreted as UTC+08:00 for interface timestamps; UCI page does not explicitly state timezone",
        "feature_units": "all concentrations ug/m3; slopes ug/m3 per hour; population standard deviations ddof=0",
        "feature_window": "At origin t use t-24..t inclusive, 25 consecutive observed hours; predict t+1",
        "allowed_domain": "Stationary Beijing US Embassy hourly PM2.5 historical reference, 2010-2014",
        "domain_gate_is": "Explicit caller assertion plus strict time/cadence/value checks; not automatic domain identification or authentication",
        "unsupported": ["SPS30 1-second sampling or 10-second averages", "moving fireground", "personal exposure or health status", "fitness for duty", "evacuation decisions", "multi-step recursive forecast"],
        "training": "2010-2012 only; selected using validation 2013H1; calibration 2013H2; final test 2014",
        "selected_model": selected_name,
        "software": {"python": platform.python_version(), "numpy": np.__version__, "pandas": pd.__version__, "scikit_learn": sklearn.__version__, "joblib": joblib.__version__},
        "load_policy": "Load only this trusted local joblib artifact; joblib is a Python serialization format and is not suitable for untrusted uploads",
    }
    predictions = pd.DataFrame({
        "origin_timestamp": frame.iloc[indices[masks["test"]] - 1]["timestamp"].map(lambda item: item.isoformat()).to_numpy(),
        "target_timestamp": times[masks["test"]].map(lambda item: item.isoformat()).to_numpy(),
        "actual_pm25_ug_m3": y[masks["test"]], "predicted_pm25_ug_m3": test_predictions,
        "persistence_pm25_ug_m3": persistence,
        "interval_lower_ug_m3": np.maximum(0, test_predictions - radius), "interval_upper_ug_m3": test_predictions + radius,
    })
    predictions.to_csv(BASE / "data" / "pm25_test_predictions.csv", index=False)
    first_target_index = int(indices[masks["test"]][0])
    demo = [{"timestamp": row.timestamp.isoformat(), "pm25_ug_m3": float(row.pm25_ug_m3), "source_domain": DOMAIN} for row in frame.iloc[first_target_index - 25:first_target_index][["timestamp", "pm25_ug_m3"]].itertuples(index=False)]
    _json_write(BASE / "data" / "pm25_demo_history.json", demo)
    _json_write(BASE / "models" / "pm25_manifest.json", manifest)
    _json_write(BASE / "reports" / "pm25_metrics.json", metrics)
    _json_write(BASE / "sources" / "pm25_provenance.json", {
        "title": "UCI Beijing PM2.5", "dataset_id": 381, "official_page": SOURCE_URL,
        "download_link_observed_on_official_page": DOWNLOAD_URL,
        "archive_member": "PRSA_data_2010.1.1-2014.12.31.csv", "local_csv": "data/pm25_source.csv",
        "csv_sha256": SOURCE_SHA256, "citation": manifest["citation"], "license": "CC BY 4.0",
        "license_url": manifest["license_url"], "retrieved_on": "2026-10-16",
        "changes": "Original CSV preserved; PM2.5 column and calendar fields selected for derived supervised windows; weather features unused",
        "timezone_assumption": manifest["timezone_assumption"],
    })
    _make_report(metrics, predictions)
    print(json.dumps({"selected_model": selected_name, "test": metrics["test"], "model_path": str(model_path)}, ensure_ascii=False, indent=2))
    return metrics


def _unavailable(reason):
    return {"available": False, "reason": reason, "source_domain": DOMAIN, "cadence_seconds": 3600}


def _valid_number(value):
    try:
        return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and value >= 0
    except (OverflowError, TypeError, ValueError):
        return False


def predict_pm25(records, model_path=None, domain="unknown"):
    """Forecast only reviewed stationary-hourly research-domain records.

    records: ordered list of {timestamp: timezone-aware ISO8601 string,
    pm25_ug_m3: finite nonnegative number}. At least 25 consecutive hourly
    observations are needed. Optional record source_domain/source and device_id
    are checked for contradictions; the domain argument is an explicit caller
    assertion, not automatic sensor-domain detection.
    """
    if domain != DOMAIN:
        return _unavailable("domain_not_validated: only stationary UCI Beijing hourly reference data are supported; 10-second fireground streams are not supported")
    if not isinstance(records, list) or len(records) < 25:
        return _unavailable("incomplete_history: at least 25 consecutive observed hourly samples (24-hour warmup) required")
    timestamps, values = [], []
    identities = set()
    for record in records:
        if not isinstance(record, dict):
            return _unavailable("invalid_record: each sample must be an object")
        if record.get("source_domain", DOMAIN) != DOMAIN or record.get("source", DOMAIN) != DOMAIN:
            return _unavailable("source_mismatch: sample provenance contradicts the allowed research domain")
        if record.get("cadence_seconds", 3600) != 3600:
            return _unavailable("cadence_mismatch: hourly source samples are required")
        for identity_field in ("device_id", "firefighter_id"):
            if identity_field in record:
                if not isinstance(record[identity_field], str) or not record[identity_field]:
                    return _unavailable("invalid_identity")
                identities.add((identity_field, record[identity_field]))
        value, timestamp = record.get("pm25_ug_m3"), record.get("timestamp")
        if not _valid_number(value):
            return _unavailable("incomplete_or_invalid_pm25: missing, negative, nonfinite, and boolean readings are not usable")
        if not isinstance(timestamp, str):
            return _unavailable("invalid_timestamp: timezone-aware ISO8601 required")
        try:
            parsed = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                return _unavailable("invalid_timestamp: timezone missing")
            parsed = parsed.astimezone(timezone.utc)
        except (ValueError, TypeError, OverflowError):
            return _unavailable("invalid_timestamp")
        if timestamps and (parsed - timestamps[-1]).total_seconds() != 3600:
            return _unavailable("cadence_or_order_mismatch: timestamps must be strictly increasing at exact one-hour intervals with no missing hours")
        timestamps.append(parsed)
        values.append(float(value))
    for name in ("device_id", "firefighter_id"):
        if len({value for key, value in identities if key == name}) > 1:
            return _unavailable("mixed_identity: samples must be from one source series")
    try:
        target_timestamp = timestamps[-1] + timedelta(hours=1)
    except OverflowError:
        return _unavailable("invalid_timestamp: next-hour timestamp is out of range")
    path = Path(model_path) if model_path is not None else BASE / "models" / "pm25_model.joblib"
    if not path.is_file():
        return _unavailable("model_missing: run python pm25_model.py --train")
    try:
        artifact = joblib.load(path)
        if artifact.get("version") != VERSION or artifact.get("source_domain") != DOMAIN or artifact.get("cadence_seconds") != 3600 or artifact.get("feature_names") != FEATURE_NAMES or artifact.get("history_observations") != 25:
            return _unavailable("model_contract_mismatch")
        prediction = float(_predict(artifact["estimator"], _features(values[-25:]).reshape(1, -1))[0])
        radius = float(artifact["interval_radius_ug_m3"])
        if not math.isfinite(prediction) or not math.isfinite(radius) or radius < 0:
            return _unavailable("model_output_invalid")
    except (OSError, ValueError, TypeError, KeyError, AttributeError, ImportError, EOFError, IndexError, OverflowError, RuntimeError, pickle.UnpicklingError) as exc:
        return _unavailable(f"model_unavailable: {type(exc).__name__}")
    return {
        "available": True,
        "next_hour_pm25_ug_m3": prediction,
        "interval_90": {"lower_ug_m3": max(0.0, prediction - radius), "upper_ug_m3": prediction + radius, "method": "empirical calibration absolute-residual interval; no guaranteed future coverage"},
        "origin_timestamp": timestamps[-1].isoformat().replace("+00:00", "Z"),
        "target_timestamp": target_timestamp.isoformat().replace("+00:00", "Z"),
        "source_domain": DOMAIN, "cadence_seconds": 3600, "horizon_seconds": 3600,
        "model_version": VERSION, "selected_model": artifact["selected_model"],
        "scope": "ambient concentration research forecast only; not a health or evacuation decision",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true", help="Reproduce chronological training, calibration, test, and report")
    parser.add_argument("--demo", action="store_true", help="Forecast from a held-out-test example history")
    args = parser.parse_args()
    if args.train:
        train()
    if args.demo:
        records = json.loads((BASE / "data" / "pm25_demo_history.json").read_text(encoding="utf-8"))
        print(json.dumps(predict_pm25(records, domain=DOMAIN), ensure_ascii=False, indent=2))
    if not args.train and not args.demo:
        parser.print_help()


if __name__ == "__main__":
    main()
