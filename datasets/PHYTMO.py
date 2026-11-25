# PHYTMO.py
import os
import random
import numpy as np
from .QueryDataset import QueryDataset
import pandas as pd
import torch

class PHYTMO(QueryDataset):
    """
    PHYTMO Dataset Loader
    
    Filename format: [Age][SubjectID][Exercise][Leg][Performance]_[Sensor]_[RepIndex].csv
    Example: A01GAT0_1_Lshin_0.csv
        - A: Age group
        - 01: Subject ID  
        - GAT: Exercise (GAT, KFE, SQT, HAA, GIS, GHT)
        - 0: Performance (quality rating)
        - 1: Series number
        - Lshin: Sensor location (Lshin, Rshin, Lthigh, Rthigh)
        - 0: Repetition index
    """
    
    DATASET_NAME = "phytmo"  # Changed to lowercase

    def __init__(
        self,
        root="data",
        split="train",
        window_size=300,
        window_step=50,
        bg_fg=None,
        args=None,
        transforms=None,
        test_subject: list = None,
    ):
        super(PHYTMO, self).__init__(
            root=root,
            split=split,
            window_size=window_size,
            window_step=window_step,
            bg_fg=bg_fg,
            args=args,
            transforms=transforms,
            test_subject=test_subject,
        )

    def parse_filename(self, filename):
        """
        Parse PHYTMO filename format: A01GAT0_1_Lshin_0.csv
        
        Returns:
            dict with keys:
                - subject: Subject identifier (e.g., 'A01')
                - ind_label: Tuple of (exercise, sensor) for labeling
                - unique_identifier: Tuple of (subject, exercise, sensor) for grouping
        """
        # Remove .csv extension
        parts = filename.rstrip('.csv').split('_')
        
        # Need at least 3 parts: [base]_[sensor]_[rep]
        if len(parts) < 3:
            return None
        
        base = parts[0]  # e.g., 'A01GAT0'
        series = parts[1]  # e.g., '1'
        sensor = parts[2]  # e.g., 'Lshin'
        rep_idx = parts[3] if len(parts) > 3 else '0'  # e.g., '0'
        
        # Parse base: A01GAT0
        if len(base) < 6:
            return None
            
        age_group = base[0]  # 'A'
        subject_id = base[1:3]  # '01'
        exercise = base[3:6]  # 'GAT'
        performance = base[6] if len(base) > 6 else '0'  # '0'
        
        # Create identifiers
        subj = f"{age_group}{subject_id}"  # 'A01'
        
        # MODIFIED: For single-sensor datasets, use only exercise as label
        # This will be detected and handled appropriately
        ind_label = (exercise,)  # Single element tuple
        
        # unique_identifier: (subject, exercise) - removed sensor for single-sensor case
        unique_identifier = (subj, exercise)
        
        return {
            'subject': subj,
            'ind_label': ind_label,
            'unique_identifier': unique_identifier,
        }

    def get_variation(self):
        """
        Returns the index of the variation in the filename after split('_')
        For PHYTMO: A01GAT0_1_Lshin_0.csv -> split gives ['A01GAT0', '1', 'Lshin', '0']
        Sensor location is at index 2
        """
        return 2

    def get_subject(self):
        """
        Returns indices for grouping/shuffling data
        For PHYTMO unique_identifier is (subject, exercise) [sensor removed for single-sensor case]
        We want to group by subject and exercise, so return [0, 1]
        """
        return [0, 1]
    
    def get_ind_label(self):
        """
        Returns indices that define the classification label
        For PHYTMO with single sensor, ind_label is (exercise,)
        So we only need index [0] (but return as list for consistency)
        """
        return [1]  # Get exercise from unique_identifier (subject, exercise)

    def get_dataset_name(self):
        return self.DATASET_NAME

    def default_data_folders(self):
        """
        Returns path to PHYTMO segmented data folder
        Assumes data is in: data/phytmo/PHYTMO_segmented_data/
        """
        return [os.path.join(self.root, "phytmo", "PHYTMO_segmented_data")]
    
    def _process_data(self):
        """
        Override parent method to handle PHYTMO-specific sorting
        (PHYTMO uses text exercise codes like 'GAT', 'KFE' instead of 'E1', 'E2')
        """
        data_folders = self.default_data_folders()
        subjLabel_to_data = {}
        unique_indices = []

        for data_folder in data_folders:
            for root_dir, _, files in os.walk(data_folder):
                for file in files:
                    if file.endswith(".csv"):
                        filepath = os.path.join(root_dir, file)
                        identifiers = self.parse_filename(file)
                        if identifiers is None:
                            continue  # Skip files that do not match expected pattern

                        subj = identifiers["subject"]
                        ind_label = identifiers["ind_label"]
                        unique_identifier = identifiers["unique_identifier"]

                        if unique_identifier not in subjLabel_to_data:
                            subjLabel_to_data[unique_identifier] = []
                        subjLabel_to_data[unique_identifier].append(filepath)

                        if ind_label not in unique_indices:
                            unique_indices.append(ind_label)
        
        # PHYTMO-specific sorting: sort by exercise name alphabetically
        # For single-sensor case, ind_label is a single-element tuple (exercise,)
        unique_indices = sorted(unique_indices, key=lambda x: x[0])

        # Proceed with splitting data into train/test and saving
        from utilities import seed
        seed(self.args.dataset_seed)
        
        import random
        train_data = {}
        train_label = {}
        test_data = {}
        test_label = {}
        for k, v in subjLabel_to_data.items():
            random_indices = random.sample(range(len(v)), int(0.2 * len(v)))
            subj = tuple([k[i] for i in self.get_subject()])
            ind_label = tuple([k[i] for i in self.get_ind_label()])
            if subj not in train_data:
                train_data[subj] = []
                train_label[subj] = []
                test_data[subj] = []
                test_label[subj] = []
            train_data[subj].extend(
                [v[i] for i in range(len(v)) if i not in random_indices]
            )
            train_label[subj].extend(
                [
                    unique_indices.index(ind_label)
                    for i in range(len(v))
                    if i not in random_indices
                ]
            )
            test_data[subj].extend(
                [v[i] for i in range(len(v)) if i in random_indices]
            )
            test_label[subj].extend(
                [
                    unique_indices.index(ind_label)
                    for i in range(len(v))
                    if i in random_indices
                ]
            )

        train = {
            "data": train_data,
            "label": train_label,
            "unique_indices": unique_indices,
            "info": "Processed data",
        }

        test = {
            "data": test_data,
            "label": test_label,
            "unique_indices": unique_indices,
            "info": "Processed data",
        }

        # Save processed data
        np.save(
            os.path.join(self.root, self.DATASET_NAME, "train_data.npy"),
            train,
            allow_pickle=True,
        )
        np.save(
            os.path.join(self.root, self.DATASET_NAME, "test_data.npy"),
            test,
            allow_pickle=True,
        )

    def data_correspondence(self):
        """
        Maps exercise labels to variation labels
        
        For single-sensor PHYTMO (only Lshin):
        Exercises (6 total):
            0: GAT (Natural Gait)
            1: GHT (Heel-Tiptoe Gait)
            2: GIS (Infinity Gait)
            3: HAA (Hip Abduction)
            4: KFE (Knee Flex-Extension)
            5: SQT (Squats)
        
        Since we only have one sensor, each exercise maps to itself.
        """
        # If noise is added, labels shift by +1
        offset = 1 if self.args.add_side_noise else 0
        
        # For single sensor, each exercise only has one variant (itself)
        dic = {
            0: [0 + offset],  # GAT
            1: [1 + offset],  # GHT
            2: [2 + offset],  # GIS
            3: [3 + offset],  # HAA
            4: [4 + offset],  # KFE
            5: [5 + offset],  # SQT
        }
        
        return dic
    
    @staticmethod
    def add_args(parser):
        """Add PHYTMO-specific arguments if needed"""
        # Currently no PHYTMO-specific args, but keep for future use
        return parser