from torchtext import data
import glob

class Corpus:
    def __init__(self, fields_examples=None, dataset_fields=None):
        """Create a Corpus by specifying examples and fields.
        Arguments:
            fields_examples: A list of lists of field values per example.
            dataset_fields: A list of pairs (field name, field object).
        Both lists have the same size (number of fields).
        """
        self.fields_examples = (
            fields_examples if fields_examples is not None else []
        )
        self.dataset_fields = (
            dataset_fields if dataset_fields is not None else []
        )
        self.number_of_examples = (
            len(self.fields_examples) if self.fields_examples else 0
        )
    
    @classmethod
    def from_files(cls, opt, fields, files):
        """Create a QualityEstimationDataset given paths and fields.
        Arguments:
            fields: A dict between field name and field object.
            files: A list with all files.
        """
        fields_examples = []
        dataset_fields = [(key, value) for key, value in fields.items()]

        for filename in files:
            pdata = cls.read_tabular_file(filename)
            if filename in glob.glob('raw_data/*/*-en/*.tsv'):
                print(filename)
                for source, target, score in zip(pdata['original'], pdata['translation'], pdata['z_mean']):
                    fields_examples.append(data.Example.fromlist([source, target, score], dataset_fields))
            elif filename in glob.glob('raw_data/*/en-*/*.tsv'): # en-de, en-zh
                print(filename)
                ndata = len(pdata['z_mean'])
                if opt.num_data:
                    ndata = int(opt.num_data*ndata)
                for source, target, score in zip(pdata['original'][0:ndata], pdata['translation'][0:ndata], pdata['z_mean'][0:ndata]):
                    fields_examples.append(data.Example.fromlist([source, target, score], dataset_fields))
        
        return cls(fields_examples, dataset_fields)
    
    @staticmethod
    def read_tabular_file(file_path, sep='\t', extract_column=None):
        
        examples = []
        line_values = []
        extract_column = None

        with open(file_path, 'r', encoding='utf8') as f:
            num_columns = None
            for line_num, line in enumerate(f):
                line = line.rstrip()
                if line:
                    values = line.split(sep)
                    line_values.append(values)
                    if num_columns is None:
                        num_columns = len(values)
                        if extract_column is not None and (
                            extract_column < 1 or extract_column > num_columns
                        ):
                            raise IndexError(
                                'Cannot extract column {} (of {})'.format(
                                    extract_column, num_columns
                                )
                            )
                    elif len(values) != num_columns:
                        raise IndexError(
                            'Number of columns ({}) in line {} is different '
                            '({}) for file: {}'.format(
                                len(values),
                                line_num + 1,
                                num_columns,
                                file_path,
                            )
                        )
                else:
                    if extract_column is not None:
                        examples.append(
                                [values[extract_column - 1] for values in line_values]
                            )
                    else:
                        examples.append(
                            [[values[i] for values in line_values] for i in range(num_columns)]
                            )

                    line_values = []
            if line_values:  # Add trailing lines before EOF.
                if extract_column is not None:
                    examples.append(
                            [values[extract_column - 1] for values in line_values]
                        )
                    
                else:
                    examples.append(
                        [[values[i] for values in line_values] for i in range(num_columns)]
                    )
        
        data = {value[0]: value[1:] for value in examples[0]}

        return data

        
