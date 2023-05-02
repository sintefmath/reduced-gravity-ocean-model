# Combining Barotropic and Baroclinic Simplified Models for Drift Trajectory Prediction

The notebooks and scripts in this repository were used to generate the results and plots in _Combining Barotropic and Baroclinic Simplified Models for Drift Trajectory Prediction_ by Florian Beiser, Havard Heitlo Holm, Martil Lilleeng Satra, Nils Melsom Kristensen, and Kai Hakon Christensen. 

## Section 3

- Figure 3.1-3.2: `notebooks/Oslofjord/Oslofjord-StaalstromParameters.ipynb`
- Table 3.1: `scripts/Oslofjord-run_time.py`

## Section 4

These simulations require that the data from the corresponding ROMS models is available, either as local netCDF-file or online.

- Figure 4.1: `notebooks/NorthCape/Baretns-SensitivityStudy.ipynb`
- Figure 4.2: `notebooks/NorthCape/Barents-BarotropicDrifters.ipynb`. It is recommended that one first runs `scripts/Barents-run_btdrift.py` and then uses the notebook for postprocessing and generation of the plots.
- Figure 4.3: `notebooks/Oslofjord/Oslofjord-MLD.ipynb`
- Figure 4.4: `notebooks/Oslofjord/Oslofjord-SensitivityStudy.ipynb`
- Figure 4.5: `notebooks/Oslofjord/Oslofjord-BaroclinicDrifters.ipynb`. It is recommended that one first runs `scripts/Oslofjord-run_bcdrift.py` and then uses the notebook for postprocessing and generation of the plots.
- Figure 4.6: `notebooks/Boknafjorden/Boknafjorden2.ipynb`
- Figure 4.7: `notebooks/Boknafjorden/Boknafjorden2-CrossProdDrifters.ipynb`. It is recommended that one first runs `scripts/Boknafjorden2-run_cpdrift.py` and then uses the notebook for postprocessing and generation of the plots.