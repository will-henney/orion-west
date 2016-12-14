files=$(find $PWD/Calibrated/BGsub -name "$1-vhel.fits")
ds9=${2:-ds9}
regprefix=Will-Regions-2016-12/pvboxes-
for path in $files; do
    f=${path##/*/} # just the filename
    if [ ${f:0:1} == "Y" ]; then
        orient=horizontal
    else
        orient=vertical
    fi
    xpaset -p $ds9 frame new
    xpaset -p $ds9 fits $path
    xpaset -p $ds9 zoom to 3 1
    xpaset -p $ds9 grid load $PWD/$orient-axes.grd
    xpaset -p $ds9 cmap bb
    xpaset -p $ds9 scale sqrt
    xpaset -p $ds9 scale limits -0.0003 0.05
    xpaset -p $ds9 contour no
    xpaset -p $ds9 contour method smooth
    xpaset -p $ds9 contour smooth 2
    xpaset -p $ds9 contour color blue
    xpaset -p $ds9 contour loadlevels $PWD/ha-contours.lev
    xpaset -p $ds9 contour yes

    regfile=${regprefix}$(basename $f .fits).reg
    if [ -f $regfile ]; then
        xpaset -p $ds9 region load $PWD/$regfile
    fi
done
xpaset -p $ds9 contour close
xpaset -p $ds9 frame next
xpaset -p $ds9 match frame wcs
xpaset -p $ds9 lock frame wcs
