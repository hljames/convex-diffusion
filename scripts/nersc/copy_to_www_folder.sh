#!/bin/bash

# list of box ids
for b in 84 85 86 87 88 89 108 109 110 111 112;
do
    echo "Copying box $b"
    cp $CFS/m3504/oisstv2-daily/subregion-60x60boxes-pixelwise_stats/sst.day.mean.box${b}.nc $CFS/m3504/www/sst.day.mean.60x60box${b}.standardized.nc
done

# list of box ids