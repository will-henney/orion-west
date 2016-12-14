import sys
import knot_fit_utils

patterns = sys.argv[1:]
if patterns:
    knot_fit_utils.process_all_slits(patterns)
else:
    knot_fit_utils.process_all_slits()
