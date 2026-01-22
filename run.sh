#!/bin/bash

rm -rf results/*
rm -rf slices/*
rm -rf paper/*
python3 ./prepareSlices.py
python3 ./extractRadiomicFeatures.py
python3 ./experiment.py
python3 ./evaluate.py
