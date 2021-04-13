# Competing AI

This code implements the competiton simulations from the [Competing AI](https://arxiv.org/pdf/2009.06797.pdf) paper published in AISTATS 2021.

To use this software, first set a configuration in the configs dir. Some examples that exactly reproduce the simulations in the paper can be found in ```configs/config_examples.py```. Note that some paths in the code may need to be changed based on the local environment. Furthermore, all of the datasets used in the paper are standard public datasets, but need to be downloaded separately. 

See the paper supplement for more details.

To run code:
```python launcher.py run -e 001EX --run-id 0```
(this will run the first simulation in configuration 001EX as defined in config.py)

Each competition instance should take between a few minutes and an hour on a single core and use less than 10 GB of memory. It is recommended to run them in parallel. Each simulation outputs some pickle files that can be loaded into memory using the routines in analysis/analysis_utils.py (see analysis/config_loading_example.ipynb for an example).

Requirements found in requirements.txt containts the exact python environment used (not every library is strictly necessary). 

Please cite our paper if you make use of these simulations in future projects:
```
@inproceedings{ginart2021competing,
  title={Competing AI: How does competition feedback affect machine learning?},
  author={Ginart, Tony and Zhang, Eva and Kwon, Yongchan and Zou, James},
  booktitle={International Conference on Artificial Intelligence and Statistics},
  pages={1693--1701},
  year={2021},
  organization={PMLR}
}
