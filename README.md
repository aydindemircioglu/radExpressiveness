
## Limitations of radiomic features: a study on a simple pattern

This is the code for 'Limitations of radiomic features: a study on a simple pattern'.


### Create virtual environment

To run the code, first create a virtual environment and install the prerequisites.

```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Then, download the WORC database (https://github.com/MStarmans91/WORCDatabase)
and put the files for CRLM, Desmoid, GIST, and Lipo into ./data/radDB/<patient-ID>,
e.g.

```
data/radDB/
├── CRLM-001/
│   ├── image.nii.gz
│   ├── segmentation_lesion0_CNN.nii.gz
│   ├── segmentation_lesion0_PhD.nii.gz
│   ├── segmentation_lesion0_RAD.nii.gz
│   ├── segmentation_lesion0_STUD1.nii.gz
│   └── segmentation_lesion0_STUD2.nii.gz
├── CRLM-002/
│   ├── image.nii.gz
│   ├── segmentation_lesion0_CNN.nii.gz
│   ├── segmentation_lesion0_PhD.nii.gz
│   ├── segmentation_lesion0_RAD.nii.gz
│   ├── segmentation_lesion0_STUD1.nii.gz
│   └── segmentation_lesion0_STUD2.nii.gz
```


### Running the experiment

You can just run ```./run.sh```  to execute all steps outlined below. 


### Prepare datasets

To extract image slices, run the following command:

```python3 ./prepareSlices.py```

This will generate all slices (with the emojis) in ./slices

For the dataset, scans were removed if one of the two slices above or below the
slice with the largest ROI did not contain any segmentation.  


### Prepare radiomic features

Since radiomic features do not depend on the train/test set, they can be
extracted upfront. For this execute

```python3 ./extractRadiomicFeatures.py```


### Experiments

The main experiment can be executed via ```python3 ./experiment.py```

Both will create the results in the ```./results``` folder.



### Evaluation

For evaluation, call ```python3 ./evaluate.py```
For the figures on the different patterns, execute ```python3 ./prepareFigureSlices.py``` INSIDE
the figures folder. DO NOT EXECUTE outside, else your ./slices will be wiped.



## LICENCE

The license for this code is MIT:

Copyright 2025, Aydin Demircioglu

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
