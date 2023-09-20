import os

from utils.sequence import Sequence


class Dataset:

    def __init__(self, workspace_path):
        
        self.sequences_dir = os.path.join(workspace_path, 'sequences')
        if not os.path.isdir(self.sequences_dir):
            print('Workspace directors (%s) does not have sequences directory.' % self.sequences_dir)
            exit(-1)

        self.sequences = []
        self._load_sequences(os.path.join(self.sequences_dir, 'list.txt'))

    @property
    def number_frames(self):
        return self.total_length

    def _load_sequences(self, list_path):
        sequence_names = self._read_sequences_list(list_path)
        self.total_length = 0
        for sequence in sequence_names:
            self.sequences.append(self._load_sequence(sequence))
            self.total_length += self.sequences[-1].length

    def _read_sequences_list(self, list_path):
        with open(list_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

    def _load_sequence(self, sequence_name):
        return Sequence(self.sequences_dir, sequence_name)
