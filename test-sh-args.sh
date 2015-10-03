files=$(find $PWD/Calibrated/BGsub -name "$1-vhel.fits")
for f in $files; do
    ls -l $f
done
