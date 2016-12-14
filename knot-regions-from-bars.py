import os
import boxbar_utils

REGION_DIR = 'Will-Regions-2016-12'
bar_pattern = 'bars-from-boxes-{}-groups.reg'
knot_pattern = 'knots-{}.reg'
for group in 'slow', 'fast', 'ultra':
    bar_file = os.path.join(REGION_DIR, bar_pattern.format(group))
    knot_file = os.path.join(REGION_DIR, knot_pattern.format(group))
    boxbar_utils.convert_bars_to_knots(bar_file, knot_file)
