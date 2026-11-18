# EDA findings (Phase 1): raw material for your own report text

All numbers come from `data/processed/*.csv` (train 80 + student_test 10 trips) or, where marked
"train raw", from the 80 training trip folders. Nothing from the evaluator partition is used.
Regenerate figures with `python scripts/make_eda_figures.py`. Write the report prose in your own
words: the guidelines check that you can explain every observation.

## Dataset inventory
- 90 trips: 80 train (208,417 GPS points, median 1,529 per trip, range 361-19,313) and 10 student_test
  (19,160 points, median 1,420, range 262-3,958). Every trip has .track, .nodes, .arcs and .route files.
- Trips are spread across the world (longitude -100 to 137, latitude -35 to 62), so each trip is projected
  to its own local metric plane (`src/data/geo.py`). Typical trip extent is about 7 x 5 km (max 70 x 75 km).

## Sampling and time gaps
- Train raw: 201,051 of 208,337 steps (96.5%) are exactly 1 s apart, matching the "about 97%" in the dataset note.
  Per trip the share ranges from 53% to 100% (mean 94.3%).
- 47 of 90 trips contain at least one gap above 5 s (1,260 gaps in total). Trip 00000004 has the most (320).
- Two trips have one huge timestamp jump: 00000090 (153,064 s, at the end) and 00000078 (52,810 s, at the start),
  so their duration and mean speed are meaningless.

## Speed outliers
- Train raw: 156 steps exceed 250 km/h. They concentrate in trip 00000013 (95 steps) and 00000032 (40 steps);
  nine more trips have 1-3 such steps (max 649 km/h in 00000079). These look like GPS jumps, not driving.

## Road networks
- No isolated nodes and no arcs pointing at missing nodes in any trip.
- 89 of 90 networks have more than one weakly connected component (median 19, max 421), but the largest
  component holds a median 98% of the used nodes (minimum 69%).
- Two-way arc share: median 95%, minimum 35%. 32 trips contain zero-length arcs; 9 trips have self-loop arcs (24 in total).
- Network size ranges from 299 to 155,003 nodes (649 to 307,392 arcs) per trip.

## Reference routes
- Consecutive route arcs are not always connected: 11 trips have a break (14 breaks in total), so the reference
  is not perfectly contiguous. Under the monotone alignment the route arcs that receive at least one GPS
  point are a median 92% (minimum 67%).
- Median GPS-to-reference distance is 4.7 m (median over trips); the median of the per-trip 95th percentile is 14.5 m.

## Case studies: trips 00000033 and 00000054 (flagged usable = 0)
- 71.1% (trip 33) and 90.6% (trip 54) of GPS points are more than 30 m from the reference route.
- Not an ordering artefact: the nearest route arc ignoring order gives the same distances
  (median 122 m and 139 m), and reversing the route is worse.
- The far points are concentrated: the five busiest 60 m x 60 m cells hold 98% (trip 33) and 96% (trip 54) of them;
  the two busiest cells hold 2,958 and 847 points (trip 33) and 7,264 and 5,094 points (trip 54).
  So these are dense clusters of GPS points, mostly 120-140 m from every arc of the reference route.
- From the data alone we cannot tell whether the vehicle was loitering off the reference route or whether the
  route omits roads it drove. Either way the labels are unreliable there, which is why the build marks both trips unusable.
  This is exactly the "reference route limitations" point the dataset note asks you to discuss.

## Assumptions and decisions to state in the report
- Local equirectangular projection per trip (error well below 0.1% at this scale).
- Candidate radius 50 m, max 10 candidates; labels from a monotone alignment of GPS to the route; a label is
  low-confidence above 30 m; a trip is unusable if more than 50% of its labels are low-confidence.
