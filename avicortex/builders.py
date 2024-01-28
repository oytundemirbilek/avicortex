"""Utility module for graph construction functions."""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT_PATH = os.path.dirname(__file__)


class GraphBuilder:
    """
    Base class for common functionalities of all datasets.

    Examples
    --------
    >>> fs_path = os.path.join(ROOT_PATH, "datasets", "my_freesurfer_stats.csv")
    >>> gbuilder = GraphBuilder(fs_path)
    >>> nodes, edges = gbuilder.construct(hem="left")
    >>> labels = gbuilder.get_labels()
    """

    def __init__(self, fs_out_path: str) -> None:
        self.fs_out_path = fs_out_path
        # Read freesurfer output table.
        self.freesurfer_df = pd.read_csv(fs_out_path)
        # Pattern used to name columns, usually: hemisphere_region_view
        self.column_pattern = r"{}_.*_{}$"
        self.region_name_pos_in_pattern = 1
        # All regions defined in DKT-Atlas
        regions_path = os.path.join(ROOT_PATH, "data", "region_names.csv")
        self.dkt_regions = pd.read_csv(regions_path)["Alias_HCP"]
        # Define view names to look for.
        self.views = ["meancurv", "gauscurv", "thickness", "area", "volume"]
        self.hemispheres = "lh", "rh"
        self.label_encoding = {"M": 0, "F": 1}

    def check_regions(self, single_view: pd.DataFrame) -> list[str]:
        """
        Check if any column does not indicate a DKT-Atlas region. Return the column names that does not.

        Parameters
        ----------
        single_view: pandas Dataframe
            Part of freesurfer dataframe that includes columns for only one view.

        Returns
        -------
        list of strings
            column names that are not a region of the DKT-atlas.
        """
        regions = self.dkt_regions.str.lower().values
        # found_regions = [
        #     c.lower().split("_")[self.region_name_pos_in_pattern]
        #     for c in single_view.columns
        # ]
        # Ensure all regions found in table are DKT-Atlas regions.
        # is_not_dkt_region = list(
        #     map(
        #         lambda c: c.lower().split("_")[self.region_name_pos_in_pattern] not in regions,
        #         single_view.columns,
        #     )
        # )
        is_not_dkt_region = [
            col.lower().split("_")[self.region_name_pos_in_pattern] not in regions
            for col in single_view.columns
        ]
        # missing_regions = list(set(regions) - set(found_regions))
        # false_regions = list(set(found_regions) - set(regions))
        # print(missing_regions)
        # print(false_regions)
        # return false_regions
        return single_view.columns[is_not_dkt_region]

    def get_labels(self) -> np.ndarray:
        """
        Get labels from freesurfer output table.

        Returns
        -------
        labels: numpy ndarray
            Labels encoded as integers in a numpy array.
        """
        return self.freesurfer_df["Gender"].map(self.label_encoding).values

    def get_subject_ids(self) -> pd.Series:
        """
        Get subject ids from freesurfer output table.

        Returns
        -------
        subject_ids: pandas Series
            Pandas column that subject IDs are located.
        """
        return self.freesurfer_df["Subject ID"]

    @staticmethod
    def anti_vectorize(vector: np.ndarray, n_nodes: int) -> np.ndarray:
        """
        Create an adjacency matrix from a given vector of lower triangular matrix.

        Parameters
        ----------
        vector: numpy ndarray
            Vectorized edges of an undirected graph. Shaped (n_nodes * n_nodes)
        n_nodes: int
            Number of nodes should be in the graph.

        Returns
        -------
        numpy ndarray
            Adjacency matrix of a single graph.
        """
        adj_matrix = np.zeros((n_nodes, n_nodes))
        adj_matrix[np.triu_indices(n_nodes, k=1)] = vector
        adj_matrix = adj_matrix.transpose()
        adj_matrix[np.triu_indices(n_nodes, k=1)] = vector
        return adj_matrix

    @staticmethod
    def vectorize(adj_matrix: np.ndarray) -> np.ndarray:
        """
        Return the lower triangular matrix as a vector of a given adjacency matrix.

        Parameters
        ----------
        adj_matrix: numpy ndarray
            A symmetric adjacency matrix of a fully connected graph. Shaped (n_nodes, n_nodes).

        Returns
        -------
        numpy ndarray
            Vectorized lower triangular matrix of the graph. Shaped (n_nodes * (n_nodes - 1) / 2).
        """
        return np.tril(adj_matrix, k=-1)

    def build_node_features(
        self, hemisphere: str, normalize: bool = True
    ) -> np.ndarray:
        """
        Extract and creates node features for each view from the freesurfer output table.

        Parameters
        ----------
        hemisphere: string
            For which hemisphere to get freesurfer data.

        Returns
        -------
        numpy ndarray
            Node features in a numpy array for all views. Shaped (n_views, n_nodes)
        """
        all_views = []
        for view in self.views:
            # Find all columns related to the view:
            pattern = self.column_pattern.format(hemisphere, view)
            df_single_view = self.freesurfer_df.filter(regex=pattern)
            # Check for erroneous columns that does not indicate a DKT-atlas region.
            not_dkt = self.check_regions(df_single_view)
            df_single_view = df_single_view.drop(not_dkt, axis="columns").values
            if normalize:
                df_single_view = df_single_view / df_single_view.max()
            # Append single view values
            all_views.append(df_single_view)
        return np.array(all_views)

    @staticmethod
    def edge_function(src_nodes: np.ndarray, dest_nodes: np.ndarray) -> np.ndarray:
        """Define how edges should be calculated. Currently pairwise absolute difference."""
        return np.abs(src_nodes - dest_nodes) / (src_nodes + dest_nodes)

    def build_edge_features(self, node_features: np.ndarray) -> np.ndarray:
        """
        Extract and creates edge features for each view from node features.

        Relative values between 2 brain regions are considered as morphological.
        So this function is to create edges represents brain morphology.

        Parameters
        ----------
        node_features: numpy ndarray
            node features array ideally calculated with build_node_features method in this class.

        Returns
        -------
        numpy ndarray
            Vectorized edge feature array shaped (n_nodes * n_nodes, n_views)
        """
        pair_node_features = np.expand_dims(node_features, axis=1)
        return self.edge_function(pair_node_features, node_features)

    def construct_for_hemisphere(
        self, hem: str, supress_node_features: bool = True, save: bool = False
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Construct graph for the given hemisphere.

        Parameters
        ----------
        hem: string
            Hemisphere to run the processing for. Can be only 'left' or 'right'.
        supress_node_features: bool
            If we do not want to encode node features in our graphs, we just fill it by 1's.
        save: bool, default False
            Whether to save the constructed graphs or not.

        Returns
        -------
        numpy ndarray
            Node features matrix including all views.
        numpy ndarray
            Edge features matrix including all views.
        """
        nodes = self.build_node_features(hem).transpose(2, 1, 0)
        edges = self.build_edge_features(nodes)
        if supress_node_features:
            nodes = np.ones_like(nodes)
        if save:
            out_path = self.fs_out_path.strip("vsc.")
            np.save(f"{out_path}_{hem}_nodes.npy", nodes)
            np.save(f"{out_path}_{hem}_edges.npy", edges)

        print(f"Hem:{hem}, Edges found:", edges.shape)
        print(f"Hem:{hem}, Nodes found:", nodes.shape)
        return nodes, edges

    def construct(
        self, save: bool = False, hem: str = "both"
    ) -> tuple[np.ndarray, ...]:
        """
        Construct graph for the given hemisphere or both.

        Parameters
        ----------
        hem: string
            Hemisphere to run the processing for. Can be only 'left', 'right' or 'both'.
        save: bool, default False
            Whether to save the constructed graphs or not.

        Returns
        -------
        numpy ndarray
            Node features matrix including all views.
        numpy ndarray
            Edge features matrix including all views.
        """
        if hem == "both":
            nodes_left, edges_left = self.construct_for_hemisphere(
                self.hemispheres[0], save
            )
            nodes_right, edges_right = self.construct_for_hemisphere(
                self.hemispheres[1], save
            )
            return nodes_left, edges_left, nodes_right, edges_right
        elif hem == "left":
            return self.construct_for_hemisphere(self.hemispheres[0], save)
        elif hem == "right":
            return self.construct_for_hemisphere(self.hemispheres[1], save)
        else:
            raise NotImplementedError("Hemisphere should be 'left', 'right' or 'both'")


class HCPGraphBuilder(GraphBuilder):
    """GraphBuilder for HCP Young Adult dataset."""

    def __init__(self, fs_out_path: str) -> None:
        super().__init__(fs_out_path)
        # Overload pattern.
        self.column_pattern = r"FS_{}_.*_{}$"
        # Overload hemisphere abbreviations.
        self.hemispheres = "L", "R"
        # Overload view names.
        self.views = ["MeanCurv", "GausCurv", "Thck", "Area", "GrayVol"]
        self.region_name_pos_in_pattern = 2

    def get_subject_ids(self) -> pd.Series:
        """
        Get subject ids from freesurfer output table.

        Returns
        -------
        pandas Series
            Pandas column that subject IDs are located.
        """
        return self.freesurfer_df["Subject"].astype("str")


class OpenNeuroGraphBuilder(GraphBuilder):
    """GraphBuilder for Openneuro Cannabis Users dataset baseline and followup."""

    def __init__(self, fs_out_path: str, include_all: bool = False) -> None:
        super().__init__(fs_out_path)
        # Overload label look up dictionary.
        self.label_encoding = {"HC": 0, "CB": 1}
        # OpenNeuro dataset comes with a metadata content which also includes labels.
        if include_all:
            meta_path = os.path.join(
                ROOT_PATH, "data", "openneuro_participants_all_timepoints.csv"
            )
        else:
            meta_path = os.path.join(ROOT_PATH, "data", "openneuro_participants.csv")
        self.meta_data = pd.read_csv(meta_path)

    def get_labels(self) -> np.ndarray:
        """
        Get labels from freesurfer output table.

        Returns
        -------
        numpy ndarray
            Labels encoded as integers in a numpy array.
        """
        return self.meta_data["group"].map(self.label_encoding).values


class CandiShareGraphBuilder(GraphBuilder):
    """GraphBuilder for Candi Share Schizophrenia Bulletin 2008 dataset."""

    def __init__(self, fs_out_path: str) -> None:
        super().__init__(fs_out_path)
        # Overload label look up dictionary.
        self.label_encoding = {"HC": 0, "BPDwoPsy": 1, "BPDwPsy": 2, "SS": 3}

    def get_labels(self) -> np.ndarray:
        """
        Get labels from freesurfer output table.

        Returns
        -------
        numpy ndarray
            Labels encoded as integers in a numpy array.
        """
        # In Candishare dataset, labels are the first part of the subject id.
        subjects = self.get_subject_ids()
        disease_names = subjects.str.split("_", n=1, expand=True)[0]
        return disease_names.map(self.label_encoding).values
