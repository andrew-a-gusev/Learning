awk 'BEGIN{print "ImageId,Label"; id=1} //{for(i=2;i<=NF;i++) {print id","$i;id++}}' prediction.log | sed 's/"\([0-9]\)"/\1/'
