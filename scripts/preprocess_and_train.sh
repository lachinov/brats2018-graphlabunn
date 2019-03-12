#!/bin/bash


term_handler()
{
        echo "Function term_handler called.  Exiting..."

        kill -TERM "$pid2" 2>/dev/null
        kill -TERM "$pid1" 2>/dev/null

        exit 255
}

trap term_handler TERM
trap term_handler INT


echo "script initiated"
echo "running augmentation"

python augment_dataset.py
augmentation_code=$?

if [ "$augmentation_code" -ne "0" ]; then
    return $augmentation_code
fi

echo "augmentation has been finished"
echo "running training"

python train.py &
pid1=$!

echo "sleeping"
sleep 20s

echo "running tensorboard"
tensorboard --logdir="../models/only_seg/logs/" --port=$1 &
pid2=$!

echo "waiting"
wait $pid1

echo "killing tensorboard"
kill $pid2

echo "finishing"