DS="covid1"
SHAPE="-w 256 -h 256 -z 1 -c 4"
LR=0.001
EPOCHS=16
IN_BS=50

#for MODEL in "generic_mlp" "generic_vgg16" "generic_vgg16_bn" "generic_resnet" "generic_resnet50_da_bn"
for MODEL in "generic_mlp" 
do

echo Modelo: $MODEL
OUT="$DS$MODEL.out"
echo Resultados en: $OUT

# Sequential
echo "========================================="
echo "SEQUENTIAL"
echo "========================================="

rm $OUT
cmd="./bin/$MODEL -m $DS $SHAPE -l $LR -e $EPOCHS -b $IN_BS"
$cmd >> $OUT



# Parallel
MODELDISTR="${MODEL}_distr"

echo Modelo: $MODELDISTR
OUT="$DS$MODELDISTR.out"
echo Resultados en: $OUT


echo "========================================="
echo "DISTRIBUTED"
echo "========================================="

rm $OUT
for n in 2 4 6 8 9 
do      
 	let BS=$IN_BS*$n
	mpirun -np $n -hostfile ../cluster.altec --map-by node ./bin/$MODELDISTR -m $DS $SHAPE -l $LR -e $EPOCHS -b $BS >> $OUT
done

done
