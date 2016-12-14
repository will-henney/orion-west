files=$(find $PWD/Calibrated/BGsub -name "$1-vhel.fits")
regdir=Alba-Regions-2016-10/blue_knots_final-SLITS
for path in $files; do
    f=${path##/*/} # just the filename
    regfile=$regdir/$(basename $f .fits).reg
    ls -l $regfile
done
