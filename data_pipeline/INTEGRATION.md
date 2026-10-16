# Connecting the sensor explorer to the data pipeline

The root `index.html`, `sensor.js`, and `style.css` remain the standalone sensor explorer. This module adds a local ingestion API and offline research models; it does not silently turn the explorer into live telemetry.

## Contract mapping

| Explorer field | Pipeline field | Required adapter behavior |
|---|---|---|
| `pm25_ug_m3` | `pm25_ug_m3` | Preserve µg/m³ and missing values. Only describe a value as an interval average if the device/gateway actually computes that average. Never substitute zero for a missing reading. |
| `timestamp` | `sample_start_utc`, `sample_end_utc` | First establish whether `timestamp` marks a sample instant, aggregation start, or aggregation end. Do not infer a 10-second interval merely because 10 seconds is the design target. An instantaneous sample must be aggregated using known collection timestamps and a documented averaging policy before it can enter this interval-average contract. |
| `latitude`, `longitude` | `latitude_deg`, `longitude_deg` | Preserve degrees and add `location_time_utc` and `location_accuracy_m` when known; otherwise use null. A phone location does not imply a sensor-measured location or position accuracy. |
| `device_id`, `firefighter_id` | Same names | Preserve pseudonymous identifiers. Do not merge multiple devices or people. |
| No field yet | `session_id` | The gateway assigns a fresh identifier per work assignment/session. Do not infer a session solely from an identifier or date. |
| `source` | `source` | Map genuine device readings to `sensor`, generated examples to `simulated`, and historical source replays to `replay`. Never relabel simulation as sensor data. |
| `quality` (`prototype`, `stale`, `invalid`) | `quality_status` (`valid`, `warming_up`, `sensor_error`, `out_of_range`) | These vocabularies are not equivalent. `prototype` is a development label and must not become `valid` automatically. `stale` concerns age and is determined from preserved sample timestamps. Map `invalid` to a specific sensor failure state only with evidence. If quality cannot be established, quarantine the record for review instead of fabricating a valid state. Preserve the original label as additional metadata if needed. |
| No field yet | `received_at_utc` | Stamp the actual gateway receipt time; never overwrite collection timestamps with receipt time after buffering. |
| No field yet | `battery_percent` | Supply measured battery percentage or null. |
| Optional `co_ppm`, `radiation_uSv_h` | Extra fields, currently unused | Preserve only if supplied by separate suitable sensors. Neither the PM2.5 pipeline nor the research models uses these fields to make health or evacuation decisions. |

The expected 1-second sampling and 10-second aggregation cadences are design targets. Confirm actual averaging, timestamp semantics, units, clock synchronization, and device-quality flags with the hardware team. The pipeline rejects duplicate/overlapping intervals, does not fill missing time, and flags deviations from the configured aggregation interval while retaining mathematically valid observed exposure.

## Local endpoints

Run `python server.py` from `data_pipeline/`. The service listens only at `http://127.0.0.1:8766`.

- `GET /health`: service availability and unavailable clinical-decision capabilities.
- `POST /v1/readings`: one contract object, or `{ "records": [object, ...] }`; use `Content-Type: application/json`.
- `GET /v1/crew/<firefighter_id>?session_id=<session_id>`: source-isolated latest status, with live sensor streams taking priority over replay/simulation.

The service rejects cross-origin browser requests. A hosted explorer cannot directly access the local service as a production integration. A reviewed same-origin backend, authenticated gateway, and deployment plan would be separate work. Do not open this prototype publicly or place location/health records in Git.

## UI meanings

`monitoring` means observation is ongoing, not that a firefighter is healthy or cleared to continue. `insufficient_data` means the information is not usable enough for the processing workflow. `check_in` requests human review. `urgent_assistance` is driven by an explicit help/serious-symptom report and remains latched during the session; it is not automatically cleared by later low PM2.5 readings.

Always treat `evacuation_required: null` as unknown, never false. `health_status` is `not_assessed` or `reported_concern`; the current hardware cannot supply the heart rate, core temperature, and skin temperature required by the physiology research model. The hourly Beijing forecast is not compatible with ten-second wearable data. Show source and data quality alongside any displayed status.
