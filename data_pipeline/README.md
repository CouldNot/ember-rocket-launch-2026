# EMBER data pipeline and research models
2026-10-16 | Verified with Python 3.10.11

Start with reports/project_report.html. This package includes models trained on real public datasets, data-cleaning and feature code, held-out test results, and a runnable data interface.

## Model and pipeline decisions
1. Current hardware: SPS30 + ESP32 + phone GPS. Use an interpretable observation pipeline to calculate the observed PM2.5 exposure index, coverage, and trends.
2. Physiology research: logistic regression outperformed the random forest candidate on validation average precision. Trained weights are included. The target is early termination of a laboratory trial within the next 5 minutes, not firefighter health or evacuation.
3. The physiology model achieved 65% recall and 3.32% precision on the independent test set, indicating poor generalization. Decision: retain it as a research baseline; do not connect it to real evacuation decisions.
4. PM2.5 research: a histogram-based gradient-boosting regressor predicts next-hour concentration. Test MAE was 12.0412 µg/m³, versus 12.0374 for the persistence baseline, so it did not improve test accuracy. The hourly model cannot be applied directly to the current 10-second SPS30 data.

## Current live-data flow
SPS30 interval averages -> UTC, identity, source, and quality validation -> accumulation over non-overlapping intervals -> exposure index and coverage -> human-review status -> raw and derived records in SQLite -> HTTP queries.
Observed exposure index = sum of [interval-average PM2.5 concentration x interval duration in minutes], in µg·min/m³. This is not an inhaled dose and cannot establish remaining safe working time.

## Interface outputs
```text
status: monitoring / check_in / urgent_assistance / insufficient_data
health_status: not_assessed or reported_concern
evacuation_required: null (unknown; do not interpret as false)
withdrawal_recommendation: requires_commander_review or urgent_human_assistance
```
monitoring means ongoing observation. It does not establish health or authorize continued work.
An explicit request for help, or a report of confusion or loss of consciousness, triggers urgent human assistance. A help report within the current session is not automatically cleared by subsequent low concentration readings.
Concentration parameters for simulated sources are demonstration settings, not medical thresholds. This package does not invent concentration thresholds for health or evacuation decisions on real sensor data.

## Quick start
Run these commands from the repository's data_pipeline directory. Trained models are already included; retraining is optional.

```shell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
.venv\Scripts\python run_demo.py
.venv\Scripts\python -m unittest discover -s tests -v
.venv\Scripts\python server.py
```

run_demo.py writes both model outputs, examples of inference being refused for real-fireground data, and simulated sensor replay results to work/demo_run. Simulated examples validate the software workflow only.
The HTTP service binds to 127.0.0.1:8766 only. It is for local integration testing, not a deployed multi-user production service. For startup options, run: python server.py --help.
POST /v1/readings accepts one JSON object or {"records": [object, ...]}.
GET /v1/crew/<firefighter_id>?session_id=<session_id> returns the current status.
GET /health checks the service.
The server uses the actual receipt clock, so historical samples submitted directly will be marked stale. For historical demonstrations, use run_demo.py or the replay mode in pipeline.py.

## Input contract
See input_schema.json and the complete examples in examples/pm25_replay.jsonl.
Minimum inputs for concentration calculations: firefighter_id, session_id, device_id, source, sample_start_utc, sample_end_utc, received_at_utc, quality_status, pm25_ug_m3.
Use µg/m³ for PM2.5 and UTC timestamps ending in Z or +00:00. Add location and device fields according to the schema. Preserve missing PM2.5 as null; never replace it with zero. A missing CO sensor cannot be replaced by inferring CO from PM2.5.
Create a new session_id for each assignment. Accumulate each person's devices and sources separately. Preserve quality flags for stale, duplicate, overlapping, and missing data; do not fabricate observed exposure time.

## Unified offline research interface
```shell
python research_inference.py --kind pm25 --input data/pm25_demo_history.json --domain uci_beijing_reference
python research_inference.py --kind physiology --input data/physiology_demo_history.json --domain prospie_experiment
```

Omitting domain or passing real_fireground causes the models to refuse inference. The domain parameter is a caller assertion, not automatic environment recognition or authorization for real deployment.

Physiology inputs must contain 6 consecutive one-minute observations from the same person and trial: participant_id, trial_id, elapsed_seconds, heart_rate_bpm, core_temp_c, skin_chest_temp_c.
Core temperature in the PROSPIE study was measured rectally. Skin temperature or an unvalidated estimate cannot be substituted. This input requirement describes the original research and is not a recommendation to purchase or use that equipment in the field.
The model uses the current value, 5-minute mean, and slope for each of the three signals: 9 features in total. model_score is not a calibrated probability. Its alert refers only to the original experimental stopping label.

## Retraining
Git intentionally excludes raw archives, original workbooks, bulk source/derived CSV files, caches, SQLite files, and local logs. The training scripts download the official UCI and PROSPIE datasets when the required local source is absent, then verify the pinned source checksum. Trained weights and small example histories are included, so the demo and inference do not require a dataset download. An internet connection is needed for first-time retraining.

```shell
python pm25_model.py --train
python physiology_model.py --train
python physiology_model.py --self-test
```

Do not repeatedly tune thresholds against the test set and then describe it as an independent test. Changes to features or models require a new held-out evaluation.

## File guide
pipeline.py / server.py: live ingestion, quality checks, exposure accumulation, help reports, and HTTP endpoints
research_inference.py / run_demo.py: offline model calls and a single-command demonstration
pm25_model.py / physiology_model.py: source-data preparation, cleaning, features, splits, training, validation, testing, model persistence, and inference
models/: both sets of trained weights and model metadata
reports/: held-out test metrics, checks, plot, and English reports; start with project_report.html. Row-level predictions are regenerated by training and are not tracked.
data/: small example histories tracked in Git; raw/cleaned CSV data are downloaded or regenerated locally and ignored
sources/: tracked provenance, source checksums, citations, and license notices; original ZIP/XLSX files stay local and are ignored
tests/: software tests for quality, accumulation, help-report priority, and source isolation

## Data sources and licenses
PM2.5: Chen, S. (2015), Beijing PM2.5, UCI, DOI 10.24432/C5JS49, CC BY 4.0.
https://archive.ics.uci.edu/dataset/381/beijing+pm2+5+data
PROSPIE: Havenith, Davey, Downie, Griggs, Richmond, Loughborough University Research Repository, DOI 10.17028/rd.lboro.26076577.v1, CC BY-NC 4.0.
https://doi.org/10.17028/rd.lboro.26076577.v1

PROSPIE data and the derived example histories/model are provided under a non-commercial research license. Preserve author attribution, license terms, and modification notices. Full provenance is in sources/. Commercial use requires separate consideration of licensing and data suitability.
Model weights use Python serialization. Load only trusted weights supplied with this package and verify their checksums; do not accept arbitrary uploaded model files.

## Next stage: evaluating whether withdrawal is necessary
First, agree on outcome labels with the team and qualified operational and medical leads: requests for help, withdrawal under an established protocol, medical intervention, recovery, and event times. Withdrawal may be caused by tactical or equipment issues, so record the reason instead of treating every withdrawal as a physiological abnormality.
Collect consented, synchronized personal physiology, environmental conditions, device quality, activity load, work phase, and protective-equipment information. Record actual missingness and event times. Use authorized pseudonymous identifiers and do not publish personal movement traces.
Split by firefighter and workday, then externally validate on unseen people and fireground conditions. Report event detection, missed events, false alerts per person-hour, availability, and warning lead time. Responsible personnel must agree on validation standards and acceptable risk in advance.
Until that validation is complete, this package does not guarantee health or fitness to continue working and does not issue automatic evacuation instructions.

## Explorer integration

See [INTEGRATION.md](INTEGRATION.md) before connecting the existing sensor explorer. The existing viewer is static and is unchanged. It does not currently post records to this service. Some runtime reason strings in the inherited research implementation remain Chinese; machine-readable status/reason codes and this collaboration documentation are English.
