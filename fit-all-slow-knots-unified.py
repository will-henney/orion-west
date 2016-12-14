import sys
import knot_fit_utils

knot_fit_utils.REGION_DIR = 'Will-Regions-2016-11/will-knots-blue-slow-SLITS'
knot_fit_utils.KNOTS_DIR = 'Knot-Fits-Unified'
knot_fit_utils.STRATEGY = 'unified'

patterns = sys.argv[1:]
if patterns:
    knot_fit_utils.process_all_slits(patterns)
else:
    knot_fit_utils.process_all_slits()
