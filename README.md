# reduced-gravity-ocean-model
Notebooks for the reduced-gravity ocean model based on the `metno/gpuocean` core

## Installation 
Follow the instruction in the `metno/gpuocean` project.
Last, to set the `$PYTHONPATH` to find the source code from the `gpuocean` repository within the environment, run the following lines 
```
conda activate gpuocean_opendrift
conda-develop /path/to/gpuocean/src
```

For experiments involving OpenDrift, a separate conda envirnoment can be created from `conda_env_opendrift.yml`.


## Plots in the paper 
Fig 3.1 (a): reduced-gravity-ocean-model/notebooks/Oslofjord/Oslofjord-StaalstromParameters.ipynb > "figs/Oslofjord-bathymetry-stations.pdf"
Fig 3.1 (b): reduced-gravity-ocean-model/notebooks/Oslofjord/Oslofjord-StaalstromParameters.ipynb > "vertical_displacement.txt" 
Fig 3.2 (a): reduced-gravity-ocean-model/notebooks/Oslofjord/Oslofjord-SensitivityStudy.ipynb > "sensitivity_figs/initial_drifter_positions.pdf" 
Fig 3.2 (b): reduced-gravity-ocean-model/notebooks/Oslofjord/Oslofjord-StaalstromParameters.ipynb > "figs/drift_bt.pdf"
Fig 3.2 (c): reduced-gravity-ocean-model/notebooks/Oslofjord/Oslofjord-StaalstromParameters.ipynb > "figs/drift_combined.pdf"

Fig 4.1: reduced-gravity-ocean-model/notebooks/NorthCape/Barents-SensitivityStudy.ipynb
Fig 4.2: reduced-gravity-ocean-model/notebooks/NorthCape/Barents-BarotropicDrifters.ipynb (requires to run: reduced-gravity-ocean-model/scripts/Barents-run_btdrift.py + move the pickles-output + input the right timestamp in the notebook) 
Fig 4.3: reduced-gravity-ocean-model/notebooks/Oslofjord/Oslofjord-MLD.ipynb
Fig 4.4: reduced-gravity-ocean-model/notebooks/Oslofjord/Oslofjord-SensitivityStudy.ipynb
Fig 4.5: reduced-gravity-ocean-model/notebooks/Oslofjord/Oslofjord-BaroclinicDrifters.ipynb (requires to run: reduced-gravity-ocean-model/scripts/Oslofjord-run_bcdrift.py + move the pickles-output + input the right timestamp in the notebook) 
Fig 4.6: reduced-gravity-ocean-model/notebooks/Boknafjord/Boknafjorden2.ipynb > "boknafjorden_figs/boknafjordenDrifts.pdf"
Fig 4.7:reduced-gravity-ocean-model/notebooks/Boknafjord/Boknafjorden2-BaroclinicDrifters.ipynb (requires to run: reduced-gravity-ocean-model/scripts/Boknafjorden2-run_cpdrift.py + move the pickles-output + input the right timestamp in the notebook)


Use the following command for pdf to eps:

gs -q -dNOCACHE -dNOPAUSE -dBATCH -dSAFER -sDEVICE=eps2write -sOutputFile=${f::-4}.eps $f