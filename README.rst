.. image:: https://img.shields.io/badge/pre--commit-enabled-brightgreen?logo=pre-commit
    :target: https://github.com/pre-commit/pre-commit
    :alt: Pre-commit enabled

.. image:: https://img.shields.io/badge/license-Apache%202.0-green.svg
    :target: https://opensource.org/licenses/Apache-2.0
    :alt: License

.. image:: https://img.shields.io/badge/code%20style-black-000000.svg
    :target: https://github.com/psf/black
    :alt: Code style: black

.. image:: http://www.mypy-lang.org/static/mypy_badge.svg
    :target: http://mypy-lang.org/
    :alt: Checked with mypy

.. image:: https://img.shields.io/badge/%20imports-isort-%231674b1?style=flat&labelColor=ef8336
    :target: https://pycqa.github.io/isort/
    :alt: isort

.. image:: https://microsoft.github.io/pyright/img/pyright_badge.svg
    :target: https://microsoft.github.io/pyright/
    :alt: pyright

.. image:: https://img.shields.io/badge/python-3.9+-blue.svg
    :target: https://www.python.org/downloads/
    :alt: Python

.. image:: https://img.shields.io/badge/security-bandit-yellow.svg
    :target: https://github.com/PyCQA/bandit
    :alt: Security Status

.. image:: https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json
    :target: https://github.com/astral-sh/ruff
    :alt: Ruff

.. image:: https://github.com/oytundemirbilek/avicortex/actions/workflows/run-quality-check.yml/badge.svg
    :target: https://github.com/oytundemirbilek/avicortex/actions/workflows/run-quality-check.yml
    :alt: Pre-commit checks

.. image:: https://github.com/oytundemirbilek/avicortex/actions/workflows/run-tests.yml/badge.svg
    :target: https://github.com/oytundemirbilek/avicortex/actions/workflows/run-tests.yml
    :alt: Unittests

.. image:: https://github.com/oytundemirbilek/avicortex/actions/workflows/python-publish.yml/badge.svg
    :target: https://github.com/oytundemirbilek/avicortex/actions/workflows/python-publish.yml
    :alt: Publish package

Avicortex - Brain Connectivity Graph Builder
============================================

Python functions and classes to build connectivity graphs and datasets from Freesurfer's brain cortical measurements.

Installation
------------

Avicortex datasets are written in torch-based classes and return torch-geometric Data objects as data samples.
Therefore you need to have a compatible torch-geometric installation along with the torch installation.

::

$ pip install avicortex

Usage
-----

First, you need to have the output stats table from Freesurfer outputs. A collection of commands can be executed
from fs_utils/stats2table.sh file. In later versions, we will provide Python utilities to extract these stats tables
as well.

One class is provided for HCP Young Adult dataset but the you need to access data from:
https://www.humanconnectome.org/study/hcp-young-adult

>>> from avicortex.datasets import HCPYoungAdultDataset

After you have the table, you can point the path to your dataset as following:

>>> freesurfer_stat_path = "path/to/hcp/outputs"
>>> hcp_dataset_tr = HCPYoungAdultDataset(hemisphere="left", mode="train", freesurfer_out_path=freesurfer_stat_path)

Note that you can select a hemisphere; 'right' or 'left' and a mode for dataloading; 'train', 'validation', 'test' or 'inference',
where 'test' is for loading the unseen data after your training is finished, and use 'inference' as an in-production mode when
you do not have the labels but you want to have predictions.

Moreover, you can select a feature set for the source and target graphs that will be returned. Consider graphs in shape:
Data(x=[batch_size, n_nodes, n_features], edge_attr=[batch_size, n_edges, n_features]). After the selection, you will have:
Data(x=[batch_size, n_nodes, selection_idx], edge_attr=[batch_size, n_edges, selection_idx]). You can specify this while initializing:

>>> hcp_dataset_tr = HCPYoungAdultDataset(hemisphere="left", mode="train", freesurfer_out_path=freesurfer_stat_path, in_view_idx=0, out_view_idx=3)

Now you can put into a torch_geometric dataloader:

>>> from torch_geometric.loader import DataLoader as PygDataLoader
>>> hcp_dataloader_tr = PygDataLoader(hcp_dataset_tr)

Let's take a look at the first sample:

>>> input_graph, target_graph = next(iter(hcp_dataloader_tr))

Output input_graph and target_graph will belong to the same patient. And you can also access their medical condition as a classification label:

>>> cls_label = target_graph.y

This class might represent different conditions, such as different diseases, age groups, or gender. For example, in HCP Young Adult dataset, label
represents patient gender.

Contributing
------------

Contributions are welcome and highly appreciated. Feel free to open issues and pull requests.
