#!/bin/bash

rm *.npy
rm kubeflow_pipeline/numpy_preprocessing/*.npy
rm kubeflow_pipeline/pandas_loading/*.npz
rm kubeflow_pipeline/*.tar.gz
rm -r locusencoder/
rm *.hdf5
rm *.joblib