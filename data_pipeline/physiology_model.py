"""PROSPIE-only research model: experimental early termination within 5 minutes.

This optional module does not assess health, fitness for duty, or firefighter
evacuation. Rocket's PM2.5 hardware cannot supply its physiological inputs.
Data and derived model are distributed for non-commercial research under
CC BY-NC 4.0 with the original authors' attribution in sources/physiology*.
"""
from __future__ import annotations

import argparse
import collections
import datetime as dt
import hashlib
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parent
RANDOM_SEED = 20261016
HORIZON_SECONDS = 300
FIELDS = ("heart_rate_bpm", "core_temp_c", "skin_chest_temp_c")
FEATURE_NAMES = [f"{name}_{suffix}" for name in FIELDS for suffix in ("current", "mean_5min", "slope_per_min_5min")]
DATA_DOI = "https://doi.org/10.17028/rd.lboro.26076577.v1"
DATA_URL = "https://ndownloader.figshare.com/files/47227096"
DATA_MD5 = "cdbdb0795d3c5d2923d199466513c163"
MODEL_VERSION = "prospie-lab-5min-v1"


def _finite(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    try:
        return math.isfinite(value)
    except (OverflowError, ValueError, TypeError):
        return False


def _features(records):
    """Features use exactly the current and previous five one-minute samples."""
    if len(records) < 6:
        return None, "需要至少连续6条分钟记录，覆盖当前时点之前5分钟。"
    window = records[-6:]
    identities = {(str(r.get("participant_id")), str(r.get("trial_id"))) for r in window}
    if len(identities) != 1 or any(not r.get("participant_id") or not r.get("trial_id") for r in window):
        return None, "历史必须属于同一受试者和同一次实验。"
    times = [r.get("elapsed_seconds") for r in window]
    if not all(_finite(t) and t >= 0 for t in times):
        return None, "实验相对时间缺失或无效。"
    if any(abs(b - a - 60) > 0.01 for a, b in zip(times, times[1:])):
        return None, "最近5分钟记录必须按60秒间隔连续升序，不自动排序或填补。"
    features = {}
    for field in FIELDS:
        values = [r.get(field) for r in window]
        if not all(_finite(v) and 0 < v < 9999 for v in values):
            return None, f"缺少有效的 {field} 连续记录，拒绝预测。"
        features[field + "_current"] = float(values[-1])
        features[field + "_mean_5min"] = sum(values) / 6.0
        # Ordinary least-squares slope on x=[0,1,2,3,4,5], in units/minute.
        features[field + "_slope_per_min_5min"] = sum((i - 2.5) * value for i, value in enumerate(values)) / 17.5
    return features, None


def predict_physiology(records, model_path=None, domain="unknown"):
    """Return a latest-window research result; refuse unknown and fireground data.

    records: chronological minute records from one experiment, each containing
    participant_id, trial_id, elapsed_seconds, heart_rate_bpm, core_temp_c and
    skin_chest_temp_c. Caller must retain source provenance and must not label
    firefighter deployment data as a PROSPIE experiment to enable this model.
    """
    base = {"available": False, "domain": domain, "model_version": MODEL_VERSION,
            "endpoint": "experimental_early_stop_within_5min",
            "health_diagnosis": None, "evacuation_decision": None}
    if domain != "prospie_experiment":
        return {**base, "reason_code": "unsupported_domain", "reason": "该模型只允许PROSPIE实验数据的研究演示；不能判定真实消防员健康或是否撤离。"}
    if not isinstance(records, list) or any(not isinstance(r, dict) for r in records):
        return {**base, "reason_code": "invalid_records", "reason": "records必须是分钟历史记录列表。"}
    # Validate every supplied record's chronology, not just a convenient tail.
    if records:
        first = records[0]
        identity = (first.get("participant_id"), first.get("trial_id"))
        last_time = None
        for record in records:
            if ("source_domain" in record and record["source_domain"] != "prospie_experiment") or record.get("source") in ("real_fireground", "fireground", "real", "sensor", "live", "synthetic", "simulated"):
                return {**base, "reason_code": "conflicting_source_domain", "reason": "记录来源与PROSPIE实验域矛盾，拒绝启用研究模型。"}
            current = record.get("elapsed_seconds")
            if (record.get("participant_id"), record.get("trial_id")) != identity:
                return {**base, "reason_code": "mixed_trials", "reason": "输入混合了受试者或实验批次。"}
            if not _finite(current) or (last_time is not None and current <= last_time):
                return {**base, "reason_code": "invalid_time_order", "reason": "采样时间无效、重复或倒序；未自动修复。"}
            last_time = current
    features, error = _features(records)
    if error:
        return {**base, "reason_code": "insufficient_history", "reason": error}
    path = Path(model_path) if model_path else ROOT / "models" / "physiology_model.joblib"
    if not path.exists():
        return {**base, "reason_code": "model_missing", "reason": "研究模型文件缺失；请先完成训练。"}
    import joblib
    import pandas as pd
    package = joblib.load(path)
    if (not isinstance(package, dict) or package.get("version") != MODEL_VERSION or
            package.get("domain") != "prospie_experiment" or package.get("feature_names") != FEATURE_NAMES or
            not _finite(package.get("threshold")) or not 0 <= package["threshold"] <= 1 or "model" not in package):
        return {**base, "reason_code": "model_metadata_mismatch", "reason": "模型版本、训练域、特征定义或阈值无效，拒绝预测。"}
    score = float(package["model"].predict_proba(pd.DataFrame([features], columns=FEATURE_NAMES))[0, 1])
    if not _finite(score) or not 0 <= score <= 1:
        return {**base, "reason_code": "invalid_model_score", "reason": "模型输出不是有限有效评分，拒绝提供结果。"}
    threshold = float(package["threshold"])
    return {**base, "available": True, "reason_code": "research_result",
            "participant_id": records[-1]["participant_id"], "trial_id": records[-1]["trial_id"],
            "elapsed_seconds": records[-1]["elapsed_seconds"], "model_score": score,
            "score_is_calibrated_probability": False, "threshold": threshold,
            "research_alert": score >= threshold, "forecast_horizon_seconds": HORIZON_SECONDS,
            "features": features,
            "reason": "仅预测实验室试验未来5分钟内因热不耐受或实验温度停止条件而提前终止的研究评分；不是健康概率或撤离建议。"}


def _seconds(value):
    if isinstance(value, dt.time):
        return value.hour * 3600 + value.minute * 60 + value.second
    if isinstance(value, str):
        parts = value.split(":")
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
    raise ValueError(f"Unsupported trial time: {value!r}")


def _red(cell):
    color = cell.font.color
    return bool(color and color.type == "rgb" and str(color.rgb)[-6:].upper() == "FF0000")


def prepare_dataset(source_path):
    """Read the original workbook, preserving provenance and excluding leakage."""
    import openpyxl
    import pandas as pd
    source_path = Path(source_path)
    if hashlib.md5(source_path.read_bytes()).hexdigest() != DATA_MD5:
        raise ValueError("The source differs from the verified PROSPIE v1 workbook")
    workbook = openpyxl.load_workbook(source_path, read_only=True, data_only=True)
    groups = collections.defaultdict(list)
    counters = collections.Counter()
    red_counts = collections.Counter()
    missing_counts = collections.Counter()
    column_map = {"heart_rate_bpm": 45, "core_temp_c": 29, "skin_chest_temp_c": 33}
    for cells in workbook["Data"].iter_rows(min_row=3):
        if cells[0].value is None:
            continue
        counters["raw_rows"] += 1
        subject, condition = str(cells[3].value), str(cells[0].value)
        row = {"participant_id": subject, "trial_id": f"P{subject}_C{condition}", "condition_id": condition,
               "elapsed_seconds": _seconds(cells[1].value), "event_code": cells[2].value}
        for field, index in column_map.items():
            cell = cells[index]
            value = cell.value
            if value is None or value == 9999 or not _finite(value):
                missing_counts[field] += 1
                value = None
            if _red(cell):
                red_counts[field] += 1
                value = None
            row[field] = value
        groups[(subject, condition)].append(row)
    workbook.close()
    sample_rows, trial_rows, cleaned_rows = [], [], []
    for (subject, condition), rows in groups.items():
        if any(b["elapsed_seconds"] - a["elapsed_seconds"] != 60 for a, b in zip(rows, rows[1:])):
            raise ValueError(f"Non-minute or unordered source trial: {subject}/{condition}")
        for row in rows:
            if row["elapsed_seconds"] == 0 and row["event_code"] in (1, 2, 3):
                counters["spurious_time_zero_terminal_codes_ignored"] += 1
        terminal = [(index, row) for index, row in enumerate(rows) if row["elapsed_seconds"] > 0 and row["event_code"] in (1, 2, 3)]
        if len(terminal) != 1:
            raise ValueError(f"Ambiguous terminal event: {subject}/{condition}")
        end_index, end = terminal[0]
        counters["rows_after_terminal_removed"] += len(rows) - end_index - 1
        # No feature row at or after the terminal event is allowed.
        rows = rows[:end_index]
        end_time, event = end["elapsed_seconds"], end["event_code"]
        counters["terminal_rows_excluded"] += 1
        trial_rows.append({"participant_id": subject, "trial_id": end["trial_id"], "condition_id": condition,
                           "terminal_event_code": event, "terminal_elapsed_seconds": end_time,
                           "endpoint": "completed" if event == 1 else "thermal_intolerance" if event == 2 else "protocol_core_temperature_stop"})
        for index, row in enumerate(rows):
            cleaned_rows.append({key: value for key, value in row.items() if key != "event_code"})
            distance = end_time - row["elapsed_seconds"]
            if event == 1 and distance < HORIZON_SECONDS:
                counters["right_censored_rows_excluded"] += 1
                continue
            features, error = _features(rows[max(0, index - 5):index + 1])
            if error:
                counters["insufficient_or_missing_history_rows_excluded"] += 1
                continue
            sample_rows.append({"participant_id": subject, "trial_id": row["trial_id"],
                                "elapsed_seconds": row["elapsed_seconds"],
                                **features, "target_early_stop_5min": int(event in (2, 3) and 0 < distance <= HORIZON_SECONDS)})
    windows = pd.DataFrame(sample_rows)
    trials = pd.DataFrame(trial_rows)
    cleaned = pd.DataFrame(cleaned_rows)
    summary = {**dict(counters), "raw_participants": len({key[0] for key in groups}), "raw_trials": len(groups),
               "raw_sampling_seconds": 60, "required_feature_missing_counts": dict(missing_counts),
               "author_red_interpolated_measurements_removed": dict(red_counts),
               "trial_outcomes": {str(key): int(value) for key, value in trials["terminal_event_code"].value_counts().items()},
               "eligible_windows": len(windows), "positive_windows": int(windows["target_early_stop_5min"].sum()),
               "eligible_participants": int(windows["participant_id"].nunique()), "eligible_trials": int(windows["trial_id"].nunique()),
               "raw_early_stop_trials": int(trials["terminal_event_code"].isin([2, 3]).sum()),
               "early_stop_trials_with_eligible_positive_windows": int(windows[windows["target_early_stop_5min"] == 1]["trial_id"].nunique()),
               "red_policy": "Only source cells with explicit red FFFF0000 are considered author interpolation. Their physiological value is set missing, and any feature window touching it is excluded.",
               "target_note": "Event 2 or 3 within (current time, current time + 300 s]. Completed trials with <300 s follow-up are right-censored and excluded. No stop metadata enters features."}
    return windows, trials, cleaned, summary


def _metrics(y, scores, threshold):
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score, precision_score, recall_score, confusion_matrix
    predictions = np.asarray(scores) >= threshold
    y = np.asarray(y)
    return {"windows": len(y), "positive_windows": int(y.sum()), "positive_prevalence": float(y.mean()),
            "pr_auc_average_precision": float(average_precision_score(y, scores)),
            "roc_auc": float(roc_auc_score(y, scores)) if len(set(y)) == 2 else None,
            "recall": float(recall_score(y, predictions, zero_division=0)),
            "precision": float(precision_score(y, predictions, zero_division=0)),
            "confusion_matrix_rows_actual_columns_predicted": confusion_matrix(y, predictions, labels=[0, 1]).tolist(),
            "threshold": float(threshold)}


def train(source_path=None):
    import shutil
    import urllib.request
    import numpy as np
    import pandas as pd
    import joblib
    import sklearn
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline
    from sklearn.linear_model import LogisticRegression
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import average_precision_score, precision_recall_curve
    for folder in ("data", "models", "reports", "sources"):
        (ROOT / folder).mkdir(parents=True, exist_ok=True)
    packaged_source = ROOT / "sources" / "physiology_prospie_v1.xlsx"
    if source_path:
        if Path(source_path).resolve() != packaged_source.resolve():
            shutil.copyfile(source_path, packaged_source)
    elif not packaged_source.exists():
        urllib.request.urlretrieve(DATA_URL, packaged_source)
    windows, trials, cleaned, cleaning = prepare_dataset(packaged_source)
    subject_labels = windows.groupby("participant_id")["target_early_stop_5min"].max()
    subject_ids = subject_labels.index.to_numpy()
    train_ids, held_ids = train_test_split(subject_ids, test_size=0.4, random_state=RANDOM_SEED, stratify=subject_labels.to_numpy())
    held_labels = subject_labels.loc[held_ids].to_numpy()
    validation_ids, test_ids = train_test_split(held_ids, test_size=0.5, random_state=RANDOM_SEED, stratify=held_labels)
    ids = {"train": train_ids, "validation": validation_ids, "test": test_ids}
    parts = {name: windows[windows["participant_id"].isin(values)].copy() for name, values in ids.items()}
    if any(part["target_early_stop_5min"].nunique() < 2 for part in parts.values()):
        raise ValueError("A split has only one class; do not silently change the seed to improve test results")
    X_train = parts["train"][FEATURE_NAMES]
    y_train = parts["train"]["target_early_stop_5min"]
    X_validation = parts["validation"][FEATURE_NAMES]
    y_validation = parts["validation"]["target_early_stop_5min"]
    candidates = {
        "logistic_regression": make_pipeline(StandardScaler(), LogisticRegression(C=1.0, class_weight="balanced", solver="liblinear", max_iter=2000, random_state=RANDOM_SEED)),
        "random_forest": RandomForestClassifier(n_estimators=250, max_depth=8, min_samples_leaf=8, max_features="sqrt", class_weight="balanced_subsample", random_state=RANDOM_SEED, n_jobs=2),
    }
    validation_results, validation_scores = {}, {}
    for name, candidate in candidates.items():
        candidate.fit(X_train, y_train)
        score = candidate.predict_proba(X_validation)[:, 1]
        validation_scores[name] = score
        validation_results[name] = _metrics(y_validation, score, 0.5)
        print(json.dumps({"candidate": name, "validation": validation_results[name]}), flush=True)
    selected_name = max(validation_results, key=lambda name: validation_results[name]["pr_auc_average_precision"])
    selected_model = candidates[selected_name]
    score = validation_scores[selected_name]
    precision, recall, thresholds = precision_recall_curve(y_validation, score)
    eligible = [(float(precision[i]), float(threshold), float(recall[i])) for i, threshold in enumerate(thresholds) if recall[i] >= 0.80]
    best = max(eligible, key=lambda entry: (entry[0], entry[1]))
    threshold = best[1]
    # Model and decision threshold are frozen before test scoring. No refit.
    package = {"model": selected_model, "threshold": threshold, "feature_names": FEATURE_NAMES,
               "model_name": selected_name, "version": MODEL_VERSION, "domain": "prospie_experiment",
               "license": "CC BY-NC 4.0", "data_doi": DATA_DOI, "score_is_calibrated_probability": False}
    joblib.dump(package, ROOT / "models" / "physiology_model.joblib", compress=3)
    test = parts["test"]
    test_scores = selected_model.predict_proba(test[FEATURE_NAMES])[:, 1]
    test_metrics = _metrics(test["target_early_stop_5min"], test_scores, threshold)
    predictions = test[["participant_id", "trial_id", "elapsed_seconds", "target_early_stop_5min"]].copy()
    predictions["model_score"] = test_scores
    predictions["research_alert"] = (test_scores >= threshold).astype(int)
    positive_events = predictions[predictions["target_early_stop_5min"] == 1].groupby("trial_id")["research_alert"].max()
    test_trial_ids = set(trials[trials["participant_id"].isin(test_ids)]["trial_id"])
    test_early_trials = trials[trials["trial_id"].isin(test_trial_ids) & trials["terminal_event_code"].isin([2, 3])]
    false_episodes = 0
    for _, group in predictions.groupby("trial_id", sort=False):
        previous_false, previous_time = False, None
        for row in group.itertuples(index=False):
            false = bool(row.research_alert and row.target_early_stop_5min == 0)
            if false and (not previous_false or previous_time is None or row.elapsed_seconds - previous_time != 60):
                false_episodes += 1
            previous_false, previous_time = false, row.elapsed_seconds
    negative_hours = float((predictions["target_early_stop_5min"] == 0).sum() / 60)
    test_metrics.update({"participants": len(test_ids), "eligible_trials": int(test["trial_id"].nunique()),
                         "early_stop_trials_in_assigned_test_subjects": len(test_early_trials),
                         "early_stop_trials_with_eligible_positive_windows": len(positive_events),
                         "early_stop_trials_warned_within_5min": int(positive_events.sum()),
                         "negative_window_hours": negative_hours, "false_alert_episodes": false_episodes,
                         "false_alert_episodes_per_negative_hour": false_episodes / negative_hours if negative_hours else None,
                         "event_metric_note": "Event warning evaluated only for early-stop trials with at least one eligible pre-event 5-minute window. Missing-data trials are separately counted and are not successes."})
    splits = {name: {"participant_ids": sorted(str(value) for value in values), "participants": len(values),
                    "eligible_trials": int(parts[name]["trial_id"].nunique()), "windows": len(parts[name]),
                    "positive_windows": int(parts[name]["target_early_stop_5min"].sum())} for name, values in ids.items()}
    for name, values in ids.items():
        windows.loc[windows["participant_id"].isin(values), "split"] = name
    report = {"model_version": MODEL_VERSION, "task": "PROSPIE experimental early stop within the next 5 minutes",
              "not_validated_for": ["firefighter health diagnosis", "firefighter evacuation", "real fireground use", "PM2.5-only devices"],
              "selected_model": selected_name, "selected_by": "validation average precision only", "test_used_for_selection": False,
              "random_seed": RANDOM_SEED, "threshold_selection": "Maximize validation precision among thresholds with validation recall >= 0.80; frozen before testing.",
              "threshold": threshold, "score_is_calibrated_probability": False, "cleaning": cleaning, "splits": splits,
              "feature_names": FEATURE_NAMES, "required_future_sensors": ["heart-rate monitor", "measured rectal core temperature (dataset method)", "chest skin-temperature sensor"],
              "validation_candidates": validation_results,
              "validation_selected_threshold": _metrics(y_validation, score, threshold),
              "validation_threshold_tradeoffs": [_metrics(y_validation, score, value) for value in sorted(set([0.2, 0.5, 0.8, threshold]))],
              "test": test_metrics, "software": {"python": sys.version.split()[0], "sklearn": sklearn.__version__, "pandas": pd.__version__, "numpy": np.__version__},
              "limitations": ["Small repeated-measures laboratory cohort; grouped holdout only, no clinical or firefighter external validation.",
                              "Target combines thermal intolerance and protocol core-temperature stopping events; it is not a diagnosis or universal withdrawal criterion.",
                              "Many heart-rate values are absent. Strict complete-history requirements exclude windows and some trials, creating availability and selection limitations.",
                              "No confidence intervals or probability calibration; point estimates are unstable with few independent positive trials.",
                              "The original workbook has no absolute timestamps; all history uses minutes since trial start.",
                              "The source's author-interpolated red physiological values are removed to avoid future interpolation leakage.",
                              "This source and derived package are CC BY-NC 4.0 for non-commercial use, with attribution."]}
    windows.to_csv(ROOT / "data" / "physiology_windows.csv", index=False)
    trials.to_csv(ROOT / "data" / "physiology_trials.csv", index=False)
    cleaned.to_csv(ROOT / "data" / "physiology_history.csv", index=False)
    predictions.to_csv(ROOT / "reports" / "physiology_test_predictions.csv", index=False)
    _write_demo_case(cleaned, test)
    (ROOT / "reports" / "physiology_report.json").write_text(json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    (ROOT / "models" / "physiology_model_metadata.json").write_text(json.dumps({key: value for key, value in package.items() if key != "model"}, indent=2), encoding="utf-8")
    attribution = {"title": "PROSPIE dataset for Prediction of Core Body Temperature from Multiple Variables and The physiological strain index does not reliably identify individuals at risk of reaching a thermal tolerance limit",
                   "authors": ["George Havenith", "Sarah Davey", "Victoria Downie", "Katy Griggs", "Victoria Richmond"],
                   "repository": "Loughborough University Research Repository", "dataset_doi": DATA_DOI,
                   "download_url": DATA_URL, "source_md5": DATA_MD5,
                   "license": "Creative Commons Attribution-NonCommercial 4.0 International",
                   "license_url": "https://creativecommons.org/licenses/by-nc/4.0/",
                   "paper_doi": "https://doi.org/10.1007/s00421-021-04642-3",
                   "modifications": "Removed 9999 and author-red interpolated physiological values, ignored t=0 pseudo-completion flags, truncated after terminal events, created causal 5-minute histories and future experimental-stop labels; trained research-only models.",
                   "noncommercial_notice": "Original data and derived model/data included for non-commercial student research. Preserve authorship, source, license and modification notices. No endorsement by the original authors."}
    (ROOT / "sources" / "physiology_provenance.json").write_text(json.dumps(attribution, indent=2, ensure_ascii=False), encoding="utf-8")
    (ROOT / "sources" / "physiology_LICENSE.txt").write_text("PROSPIE original dataset and this derived research package: CC BY-NC 4.0.\nAuthors: George Havenith; Sarah Davey; Victoria Downie; Katy Griggs; Victoria Richmond.\nSource: " + DATA_DOI + "\nLicense: https://creativecommons.org/licenses/by-nc/4.0/\nLegal code: https://creativecommons.org/licenses/by-nc/4.0/legalcode\nNon-commercial use only under the license. Preserve attribution and identify modifications. No medical, safety, commercial-use or author-endorsement claim is made.\nSee physiology_provenance.json for transformations.\n", encoding="utf-8")
    _write_summary(report)
    print(json.dumps({"selected_model": selected_name, "test": test_metrics, "cleaning": cleaning}), flush=True)
    return report


def _write_summary(report):
    cleaning, test = report["cleaning"], report["test"]
    lines = ["PROSPIE 实验室研究模型说明", "", "任务：利用过去5分钟生理记录，预测未来5分钟内是否发生原始实验提前终止（Event 2 或 3）。",
             "Event 2 表示实验中热不耐受，Event 3 表示达到原研究核心温度停止条件。这不是临床诊断、健康概率或消防员撤离模型。",
             "当前 Rocket 的 SPS30 / PM2.5 数据无法提供本模型必需输入。本模块只有在 domain=prospie_experiment 时可运行。", "",
             "数据：40名参与者，155次实验，17264条分钟记录。原研究数据公开版比论文分析子集更大，应以实际清洗报告为准。",
             f"清洗：忽略 {cleaning['spurious_time_zero_terminal_codes_ignored']} 个 t=0 伪完成代码；删除终止后 {cleaning['rows_after_terminal_removed']} 行；排除 {cleaning['terminal_rows_excluded']} 个终止时点；无事件且随访不足5分钟的 {cleaning['right_censored_rows_excluded']} 行排除。",
             "9999及作者以红色标记、可能利用未来相邻点插值的生理值都置缺失，任何触及缺失值的6点特征窗都排除。原始数据仍单独保留。",
             f"可用建模窗口 {cleaning['eligible_windows']}，阳性窗口 {cleaning['positive_windows']}；这是相关的分钟窗口，不能当作独立人数。", "",
             f"完整历史要求下仅保留 {cleaning['eligible_participants']} / {cleaning['raw_participants']} 人、{cleaning['eligible_trials']} / {cleaning['raw_trials']} 次实验；{cleaning['raw_early_stop_trials']} 次提前停止中仅 {cleaning['early_stop_trials_with_eligible_positive_windows']} 次有可用阳性窗口。",
             "输入：实测心率、直肠核心温度、胸部皮温；各自的当前值、最近6个分钟点均值、过去5分钟斜率。共9个因果特征，不使用Event、停止时间、距离结束时间或实验总时长。",
             "预测历史须同一participant_id/trial_id，elapsed_seconds逐分钟递增，最近6条完整。需要不同设备或间接估计核心温度时必须重新验证。", "",
             f"模型：{report['selected_model']}，仅按验证集 average precision 选型；阈值 {report['threshold']:.6f} 仅根据验证集召回≥0.80时精确率最高的点设置。没有用测试集选型、调参或定阈值。",
             "受试者整体分配为约60/20/20训练/验证/测试，固定随机种子；同一个人的多次实验不会跨组。未做概率校准。", "",
             f"测试：{test['participants']} 人，{test['windows']} 窗口，{test['positive_windows']} 阳性窗口。",
             f"PR-AUC（average precision）={test['pr_auc_average_precision']:.4f}；ROC-AUC={test['roc_auc']:.4f}；召回率={test['recall']:.4f}；精确率={test['precision']:.4f}。",
             f"混淆矩阵（行真实0/1，列预测0/1）：{test['confusion_matrix_rows_actual_columns_predicted']}。",
             f"测试组共有 {test['early_stop_trials_in_assigned_test_subjects']} 次提前终止实验，其中 {test['early_stop_trials_with_eligible_positive_windows']} 次具备有效事件前窗口，{test['early_stop_trials_warned_within_5min']} 次在事件前5分钟内报警。",
             f"阴性观察时间 {test['negative_window_hours']:.2f} 小时，误报片段 {test['false_alert_episodes']} 次，即 {test['false_alert_episodes_per_negative_hour']:.3f} 次/阴性观察小时。连续相邻误报分钟合并为一次片段。",
             f"本次测试精确率仅 {100 * test['precision']:.2f}%，验证表现未能稳定推广到保留受试者；结果只能支持研究流程展示，不能支持个人安全决策。",
             "缺失严重的实验和窗口会被排除，报告条件性性能不能解释为全部人群的可靠性。小样本、重复测量和实验室环境均限制推广；尚未有独立消防员外部验证。", "",
             "复现：python physiology_model.py --train （首次需要原始数据下载；包内已有原始数据时不联网）。",
             "推理：predict_physiology(records, domain='prospie_experiment')。默认unknown以及现实火场均返回available:false。",
             "模型分数不是校准概率。输出health_diagnosis与evacuation_decision始终为null。", "",
             "原始数据及本衍生研究包按 CC BY-NC 4.0 非商业许可提供，必须保留作者归属与修改说明。",
             "作者：George Havenith, Sarah Davey, Victoria Downie, Katy Griggs, Victoria Richmond。",
             "数据：" + DATA_DOI, "论文：https://doi.org/10.1007/s00421-021-04642-3", "许可：https://creativecommons.org/licenses/by-nc/4.0/", "",
             "详细清洗、分组、验证阈值权衡及测试输出见 physiology_report.json 和 physiology_test_predictions.csv。"]
    (ROOT / "reports" / "physiology_summary.txt").write_text("\n".join(lines), encoding="utf-8")


def _write_demo_case(history, test):
    """First held-out positive window in source order, not chosen by model score."""
    candidates = test[test["target_early_stop_5min"] == 1]
    if candidates.empty:
        return
    sample = candidates.iloc[0]
    selected = history[(history["trial_id"] == sample["trial_id"]) &
                       (history["elapsed_seconds"] >= sample["elapsed_seconds"] - 300) &
                       (history["elapsed_seconds"] <= sample["elapsed_seconds"])]
    records = selected[["participant_id", "trial_id", "elapsed_seconds", *FIELDS]].to_dict(orient="records")
    for record in records:
        record["participant_id"] = str(record["participant_id"])
        record["trial_id"] = str(record["trial_id"])
    if len(records) != 6 or _features(records)[0] is None:
        raise ValueError("Held-out demo does not have a complete six-point history")
    (ROOT / "data" / "physiology_demo_history.json").write_text(json.dumps(records, indent=2, allow_nan=False), encoding="utf-8")
    manifest = {"source": DATA_DOI, "split": "test", "domain": "prospie_experiment",
                "selection": "First eligible positive test window in source order, selected without using model scores.",
                "actual_target_early_stop_within_5min": 1, "participant_id": records[0]["participant_id"],
                "trial_id": records[0]["trial_id"], "health_diagnosis": None, "evacuation_decision": None,
                "note": "Ground truth describes a recorded laboratory early-stop event. It is not a firefighter health or evacuation label."}
    (ROOT / "reports" / "physiology_demo_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _self_test():
    assert not predict_physiology([], domain="unknown")["available"]
    assert not predict_physiology([], domain="real_fireground")["available"]
    rows = [{"participant_id": "test", "trial_id": "one", "elapsed_seconds": index * 60,
             "heart_rate_bpm": 90 + index, "core_temp_c": 37.0 + index / 100,
             "skin_chest_temp_c": 34.0 + index / 100} for index in range(6)]
    feature, error = _features(rows)
    assert error is None and abs(feature["heart_rate_bpm_slope_per_min_5min"] - 1) < 1e-9
    assert _features(rows[:5])[0] is None
    bad = [dict(row) for row in rows]
    bad[-1]["heart_rate_bpm"] = 9999
    assert _features(bad)[0] is None
    bad[-1]["heart_rate_bpm"] = float("nan")
    assert _features(bad)[0] is None
    bad = [dict(row) for row in rows]
    bad[-1]["elapsed_seconds"] = 240
    assert not predict_physiology(bad, domain="prospie_experiment")["available"]
    bad = [dict(row) for row in rows]
    bad[-1]["trial_id"] = "other"
    assert not predict_physiology(bad, domain="prospie_experiment")["available"]
    bad = [dict(row) for row in rows]
    bad[-1]["source_domain"] = "real_fireground"
    assert predict_physiology(bad, domain="prospie_experiment")["reason_code"] == "conflicting_source_domain"
    bad = [dict(row) for row in rows]
    bad[-1]["source"] = "sensor"
    assert predict_physiology(bad, domain="prospie_experiment")["reason_code"] == "conflicting_source_domain"
    assert not any("event" in feature or "stop" in feature or "end" in feature for feature in FEATURE_NAMES)
    if (ROOT / "models" / "physiology_model.joblib").exists():
        result = predict_physiology(rows, domain="prospie_experiment")
        assert result["available"] and result["evacuation_decision"] is None and not result["score_is_calibrated_probability"]
        import joblib
        import numpy as np
        from unittest.mock import patch
        package = joblib.load(ROOT / "models" / "physiology_model.joblib")
        for key, value in (("version", "invalid"), ("domain", "real_fireground"), ("feature_names", []), ("threshold", float("nan"))):
            corrupt = dict(package)
            corrupt[key] = value
            with patch("joblib.load", return_value=corrupt):
                assert predict_physiology(rows, domain="prospie_experiment")["reason_code"] == "model_metadata_mismatch"
        class InvalidScore:
            def predict_proba(self, frame):
                return np.array([[0.0, float("nan")]])
        corrupt = dict(package)
        corrupt["model"] = InvalidScore()
        with patch("joblib.load", return_value=corrupt):
            assert predict_physiology(rows, domain="prospie_experiment")["reason_code"] == "invalid_model_score"
        print("physiology self-test: 17 checks passed")
    else:
        print("physiology input self-tests passed; model-dependent checks skipped (model not yet trained)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--source", help="Optional original PROSPIE XLSX")
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--demo", action="store_true", help="Replay the packaged six-point held-out research history")
    args = parser.parse_args()
    if args.train:
        train(args.source)
    if args.self_test:
        _self_test()
    if args.demo:
        records = json.loads((ROOT / "data" / "physiology_demo_history.json").read_text(encoding="utf-8"))
        print(json.dumps(predict_physiology(records, domain="prospie_experiment"), ensure_ascii=True, allow_nan=False))
    if not args.train and not args.self_test and not args.demo:
        parser.print_help()
