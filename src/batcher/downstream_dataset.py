import os

import numpy as np
from scipy.io import loadmat
from scipy.signal import butter, filtfilt

from batcher.base import EEGDataset


class MotorImageryDataset(EEGDataset):
    def __init__(
        self,
        filenames,
        sample_keys,
        chunk_len=500,
        num_chunks=10,
        ovlp=50,
        root_path="",
        gpt_only=True,
    ):
        super().__init__(
            filenames,
            sample_keys,
            chunk_len,
            num_chunks,
            ovlp,
            root_path=root_path,
            gpt_only=gpt_only,
        )

        self.data_all = []
        for fn in self.filenames:
            self.data_all.append(np.load(fn))

        self.mi_types = {
            769: "left",
            770: "right",
            771: "foot",
            772: "tongue",
            1023: "rejected",
        }  # , 783: 'unknown', 1023: 'rejected'
        # Types of motor imagery
        self.labels_string2int = {
            "left": 0,
            "right": 1,
            "foot": 2,
            "tongue": 3,
        }  # , 'unknown': -1
        self.Fs = 250  # 250Hz from original paper
        # Detect number of channels from first data file
        first_data = self.data_all[0]["s"]
        n_channels = (
            first_data.shape[1] if len(first_data.shape) > 1 else first_data.shape[0]
        )

        # Get absolute path to inputs directory
        script_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        inputs_dir = os.path.join(script_dir, "..", "inputs")

        # Load appropriate projection matrix based on number of channels
        if n_channels == 22:
            # BCI2A dataset with 22 channels - use original projection matrix
            self.P = np.load(os.path.join(inputs_dir, "tMatrix_value.npy"))
        elif n_channels == 19:
            # ADHD dataset with 19 channels - use interpolation matrix to map to 22
            self.P = np.load(os.path.join(inputs_dir, "tMatrix_19to22_adhd.npy"))
        else:
            # For other channel counts, use identity matrix (no transformation)
            print(
                f"Warning: Unexpected channel count {n_channels}. Using identity matrix."
            )
            self.P = np.eye(n_channels, dtype=np.float64)

        self.trials, self.labels, self.num_trials_per_sub = self.get_trials_all()
        # keys of data ['s', 'etyp', 'epos', 'edur', 'artifacts']

    def __len__(self):
        return sum(self.num_trials_per_sub)

    def __getitem__(self, idx):
        return self.preprocess_sample(
            self.trials[idx], self.num_chunks, self.labels[idx]
        )

    def map2pret(self, data):
        # data shape: (n_trials, n_channels, time_samples)
        # P shape: (22, n_channels) or (n_channels, n_channels)
        # Result: (n_trials, 22, time_samples)

        if len(data.shape) == 3:
            # Apply projection to each trial
            n_trials = data.shape[0]
            result = np.zeros((n_trials, self.P.shape[0], data.shape[2]))
            for i in range(n_trials):
                result[i] = np.matmul(
                    self.P, data[i]
                )  # (22, 19) @ (19, time) -> (22, time)
            return result
        else:
            # 2D data: (n_channels, time)
            return np.matmul(self.P, data)  # (22, 19) @ (19, time) -> (22, time)

    def get_trials_from_single_subj(self, sub_id):
        raw = self.data_all[sub_id]["s"].T
        events_type = self.data_all[sub_id]["etyp"].T
        events_position = self.data_all[sub_id]["epos"].T
        events_duration = self.data_all[sub_id]["edur"].T
        artifacts = self.data_all[sub_id]["artifacts"].T

        # Check if this is BCI2A-style data (with trial markers) or continuous data (ADHD-style)
        startrial_code = 768
        starttrial_events = events_type == startrial_code
        idxs = [i for i, x in enumerate(starttrial_events[0]) if x]

        # If no BCI2A-style trials found, treat entire recording as one trial
        if len(idxs) == 0:
            trial_labels = self.get_labels(sub_id)
            # trial_labels might be scalar or array
            if isinstance(trial_labels, (int, np.integer)):
                label = trial_labels
            else:
                label = trial_labels[0] if len(trial_labels) > 0 else 0

            trials = []
            classes = []

            # Extract the entire recording as one trial
            n_channels = raw.shape[0]
            start = events_position[0, 0]
            stop = start + events_duration[0, 0]

            # Extract full recording (or subsample if too long)
            full_trial = raw[:n_channels, start:stop]

            # Split into smaller chunks if needed for training
            # Use chunk_len from initialization or a default value
            chunk_size = 1000  # samples
            for chunk_start in range(
                0, full_trial.shape[1] - chunk_size, chunk_size // 2
            ):
                chunk = full_trial[:, chunk_start : chunk_start + chunk_size]
                if chunk.shape[1] == chunk_size:  # Only add full-sized chunks
                    trials.append(chunk)
                    classes.append(label)

            return trials, classes

        # Original BCI2A-style trial extraction
        trial_labels = self.get_labels(sub_id)

        trials = []
        classes = []
        for j, index in enumerate(idxs):
            try:
                classes.append(trial_labels[j])

                start = events_position[0, index]
                stop = start + events_duration[0, index]
                # Extract all available channels (not hardcoded to 22)
                n_channels = raw.shape[0]
                trial = raw[:n_channels, start + 500 : stop - 375]
                # add band-pass filter
                # self.bandpass_filter(trial, lowcut=4, highcut=40, fs=250, order=5)
                trials.append(trial)
            except:
                # print("Cannot load trial")
                continue
        return trials, classes

    def get_labels(self, sub_id):
        label_path = self.root_path + "true_labels/"
        base_name = os.path.basename(self.filenames[sub_id])
        sub_name = os.path.splitext(base_name)[0]
        labels = loadmat(label_path + sub_name + ".mat")["classlabel"]
        return labels.squeeze() - 1

    def get_trials_all(self):
        trials_all = []
        labels_all = []
        total_num = []
        for sub_id in range(len(self.data_all)):
            trials, labels = self.get_trials_from_single_subj(sub_id)
            total_num.append(len(trials))

            trials_all.append(np.array(trials))
            # Flatten labels in case they're returned as lists
            labels_all.extend(labels)  # Use extend instead of append to concatenate
        # reordered_data = self.reorder_channels(np.vstack(trials_all))
        trials_all_arr = np.vstack(trials_all)
        # map to same channel configuration as pretraining
        trials_all_arr = self.map2pret(trials_all_arr)
        return self.normalize(trials_all_arr), np.array(labels_all), total_num

    # def normalize(self, data):
    #     return (data - np.mean(data)) / np.std(data)

    def bandpass_filter(self, data, lowcut, highcut, fs, order=5):
        """
        Apply a bandpass filter to the data.

        Parameters:
        - data: The EEG signal
        - lowcut: Low cut-off frequency
        - highcut: High cut-off frequency
        - fs: Sampling rate (frequency)
        - order: Order of the filter

        Returns:
        - Filtered data
        """
        nyq = 0.5 * fs
        low = lowcut / nyq
        high = highcut / nyq

        b, a = butter(order, [low, high], btype="band")
        filtered_data = filtfilt(b, a, data)

        return filtered_data
