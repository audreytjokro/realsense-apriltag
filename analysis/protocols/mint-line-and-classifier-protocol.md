# Mint line, response-lag, and classifier protocol

## What is being measured

Keep these delays separate:

1. **Acquisition alignment** is the computer-time difference between a Cyranose
   readout and its nearest RealSense pose. The recorder accepts a pose only when
   this difference is at most 250 ms.
2. **Physical response dynamics** include air transport to the inlet, sensor
   reaction time, and recovery/carryover. A fixed time shift can correct a dead
   time, but it cannot completely undo a slow rise or slow recovery.

## Surface and room setup

- Use the desk as support, but do not apply essential oil directly to it.
- Prefer a clean, removable, nonporous surface such as glass, stainless steel,
  or a low-odor rigid plastic tray. A coordinate grid can sit underneath glass.
- Create the source with a narrow, replaceable absorbent strip on that surface.
  Apply a recorded volume with a pipette. If no pipette is available, use the
  same dropper and record the number of drops, but expect greater variation.
- Use an identical dry strip for blank/control trials.
- Plain paper is acceptable for a shakedown only. It absorbs and wicks oil, so
  the true source width becomes uncertain. Do not reuse it.
- Keep fans off and avoid changing doors, HVAC, surface location, pump method,
  or nearby odor sources during a comparison. Ventilate safely after testing.
- Fix the snout height with a physical guide. Start with a safe 1--2 cm gap and
  use the same value throughout. Aim for a height range no larger than 1 cm.
- Use a marked track or rail. A target speed near 1 cm/s is suitable for the
  current approximately 1.8 Hz Cyranose readout rate. Record the actual speed.

## Experiment A: physical-lag calibration

Place a straight mint source perpendicular to the scan direction:

```text
                         mint source
                             |
forward:  blank  ----------->|----------->  blank
reverse:  blank  <-----------|<-----------  blank
                         known x = x0
```

Do **not** move along the mint line. Cross it from background to background.
This supplies a known spatial event and clean observations on both sides.

1. Mark a 30 cm path with the source crossing its midpoint. Record the source
   line in desk coordinates.
2. Keep the source geometry, mint amount, height, and speed unchanged.
3. During the baseline phase, keep the inlet in clean air and do not scan.
4. During the scan phase, hold at least 5 seconds in the starting blank zone,
   traverse the 30 cm track at about 1 cm/s, then hold in the ending blank zone
   for at least 20 seconds to observe the response and recovery.
5. Perform at least three independent forward passes and three independent
   reverse passes. Let the instrument purge/recover between passes; do not
   begin the next trial merely because a timer expired.
6. Perform at least two matching dry-strip blank passes. More are better.

For equal forward and reverse speed magnitude `v`, a first lag estimate is:

```text
lag_seconds = (forward_peak_x - reverse_peak_x) / (2 * v)
source_x_estimate = (forward_peak_x + reverse_peak_x) / 2
```

The analysis should also scan candidate lags and compare the known distance to
the line with the 32-sensor response. Report the best lag separately for each
pass, plus the median and variation across passes. If different directions or
trials require very different lags, use a dynamic rise/recovery model instead
of pretending that one fixed delay explains the sensor.

## Experiment B: two-dimensional line recovery

This is different from lag calibration. Raster-scan the **whole test area**,
including substantial odor-free background on both sides of the line.

- Use parallel passes with regular spacing, such as 1--2 cm.
- Alternate directions only if the lag has already been estimated. Otherwise,
  direction-dependent response smearing will look like a spatial feature.
- Include a geometrically identical blank scan.
- Keep speed and height fixed. Reject samples with invalid pose alignment and,
  during analysis, reject samples outside the chosen height band.

The correct question is not only whether a bright line appears. Compare the
estimated ridge with the recorded ground-truth line position and report its
location error, width, contrast relative to blank, and repeatability.

## Experiment C: mint-versus-blank classifier data

Use one separate recorder session per independent trial. The combined CSV is
the preferred raw format because it contains:

- `pcnose_S1_kohm` through `pcnose_S32_kohm`;
- `pcnose_flag`, device/host time, thermistor, and raw serial frame;
- synchronization validity and latency;
- pose and snout coordinates when a valid pose is available.

Do not use concentrated bottle headspace if the final task is a dilute moving
line. Train with the same substrate, amount range, height, and sampling method
that the mapping experiment will use.

For a pilot dataset:

1. First collect two blank and two mint shakedown trials and inspect them.
2. If the phase flags, baseline, response, and recovery look valid, continue to
   at least 12 independent blank and 12 independent mint trials.
3. Each trial should contain at least 30 seconds of usable clean baseline and
   at least 30 seconds (preferably 45--60 seconds) of the intended exposure
   phase. This is **not** a 30-second total file. Use the phase flag and actual
   exposure event, not total recording duration, to decide when to stop.
4. A blank trial uses the same apparatus and motion but an identical dry strip.
5. Start a new Identify/run cycle for each trial and allow adequate recovery.
   A long continuous recording with 12 adjacent windows is not 12 independent
   trials.
6. Balance order where recovery permits. A blank after mint is also a useful
   carryover check; reject or repeat it if the baseline has not recovered.

Example blank command:

```powershell
python record_cyranose_reading_pose.py --port COM4 --baud 57600 --interval 0.2 --max-sync-ms 250 --trial-id blank_01 --trial-label blank --notes "dry strip; fixed height; static exposure"
```

Example mint command:

```powershell
python record_cyranose_reading_pose.py --port COM4 --baud 57600 --interval 0.2 --max-sync-ms 250 --trial-id mint_01 --trial-label mint --notes "mint strip; same height and method as blank_01"
```

Video is optional for stationary classifier trials. Add `--save-video` for line
scans and whenever visual quality control is useful.

## First classifier

Treat the trial, not each CSV row, as the independent example.

1. Estimate each sensor's baseline `R0` from the stable baseline phase.
2. Create a 32-sensor trial feature from the exposure phase, initially a robust
   maximum or median fractional change relative to `R0`.
3. Train a regularized logistic-regression model and output mint probability.
4. Split and evaluate by entire trials. Never randomly split adjacent rows from
   the same trial across training and testing.
5. Report cross-validated ROC-AUC, precision/recall, confusion matrix, and the
   blank false-positive rate. Reserve later-day trials for a true generalization
   check.
6. Apply the model only to alignment-valid scan readings, correct the measured
   physical response lag, and then spatially aggregate mint probability.

Twelve trials per class are enough for a pilot, not proof of generalization.
Add more trials, days, mint amounts, and non-mint negative odors based on the
learning curve and failure cases.

## Run acceptance checks

- At least 95% of Cyranose readings have accepted pose alignment at 250 ms.
- Absolute alignment p95 is reported; lower than 100 ms is an initial target.
- Snout position is valid for the scan region.
- Height range is no greater than 1 cm for the controlled experiment.
- The flag phases and exposure interval are present.
- Blank baseline has recovered before the next trial.
- Ground-truth source location, direction, height, speed, amount, and substrate
  are recorded in the trial manifest.
