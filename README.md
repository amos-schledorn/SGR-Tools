
# SGR-tools
This is a package for generating and reduction for stochastic programming problems.

Currently, it features scenario generation via seasonal block bootstrap and scenario reduction via K-Medoids clustering, adjusted for multi-stage stochastic programming problems.

## Installation
### Recommended
We recommend installing the package as editable This way, you can make changes to the code and use the package without having to reinstall it. This is especially encouraged, since the package might not be actively maintained.
#### Clone this repository
```
git clone https://github.com/amosschle/sgr-tools.git
```
#### Install the editable package using pip:
```
pip install -e sgr-tools
```

### Updates
To update the package, simply pull the latest version from the repository:
```
cd sgr-tools
git pull
```

## Citation
Please consider citing this work as:
```Amos Schledorn. (2024). SGR-Tools v0.2 (0.2). Zenodo. https://doi.org/10.5281/zenodo.11216228```


## Licence
Copyright ©2024 Technical University of Denmark.

This version of the software was developed by Amos Schledorn, postdoctoral researcher at DTU Compute.

Licensed under the [BSD 3-Clause Licence](https://github.com/amosschle/stochastic-programming-tools/blob/main/LICENSE).

## Acknowledgments
Part of this work was supported by the project PtX, Sector Coupling and LCA, which is part
of the Danish MissionGreenFuels portfolio of projects.