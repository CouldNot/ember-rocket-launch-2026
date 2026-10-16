# EMBER Sensor Explorer

Standalone interactive 3D SPS30 sensor viewer for the breathing-zone wearable. No build step or JavaScript dependencies. Open index.html in a browser, or serve this directory with any static host.

## Explore
- Drag or use arrow keys on the canvas to rotate.
- Scroll or use the +/− buttons to zoom.
- Click component geometry, numbered hotspots, or the accessible component buttons for explanations.
- Explode/assemble parts, show/hide the enclosure, and reset the view.

Only the sensor is displayed. Internal layout is illustrative, not manufacturing CAD. The page distinguishes measured PM2.5 from phone-added metadata and optional separate sensors. No live telemetry or health assessment is provided.

## Enable permanent hosting
A repository administrator can enable Settings → Pages → Deploy from a branch → main → / (root). Once built, the viewer will be at https://couldnot.github.io/ember-rocket-launch-2026/.

## Dashboard integration
Link a Sensor tab to the hosted viewer, or embed it in an iframe titled "EMBER interactive sensor explorer". The model is kept in this separate repository. Google Fonts is optional; system font fallbacks work offline.

## Source basis
- Project wearable-design.md: breathing-zone exposure wearable build specification, October 16, 2026.
- [Sensirion SPS30 datasheet](https://sensirion.com/file/datasheet_sps30), version 2.0.

## Verification
JavaScript syntax and scripted interaction checks pass for six component selections, projected hotspot picking, rotation, zoom, and enclosure/explosion toggles.
