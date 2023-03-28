#%%
# IMPORT DEPENDENCIES--------------------------------------------------------------------------------------------------
"""
Created: Monday, February 20th, 2023
Edited: Monday, March 17th, 2023
Author: Daniel Persaud
Description: This script cleans the featurized dataset and outputs a subset of the data which:
- takes the lowest-energy compound for each unique composition
- has a value for bandgap
- has a bandgap energy between 0eV/atom and 1.5eV/atom
- is a halogen
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

