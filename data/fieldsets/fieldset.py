from functools import partial
import glob

from data.vectors import AvailableVectors

class Fieldset:
    ALL = 'all'
    TRAIN = 'train'
    VALID = 'valid'
    TEST = 'test'

    def __init__(self):
        """
        """
        self._fields = {}
        self._options = {}
        self._vocab_options = {}
        self._vocab_vectors = {}

    def add(self, name, field, vocab_options=None,
        vocab_vectors=None):
        """
        Args:
            name:
            field:
            file_option_suffix:
            required (str or list or None):
            file_reader (callable): by default, uses Corpus.from_files().
        Returns:
        """
        self._fields[name] = field
        self._vocab_options[name] = vocab_options
        self._vocab_vectors[name] = vocab_vectors

    @property
    def fields(self):
        return self._fields

    def fields_and_files(self, set_name, opt, **kwargs):
        
        fields = {}
        files = []
        file_path = opt.paths.get(set_name)
        for file_name in glob.glob(file_path + '*/*.tsv'):
            if not file_name:
                raise FileNotFoundError(
                    'File {} is required.'.format(file_name)
                )
            elif file_name:
                files.append(file_name)
        fields = self._fields
        return fields, files


    # For vocabulary option loading
    def vocab_kwargs(self, name, opt):
        if name not in self._vocab_options:
            raise KeyError(
                'Field named "{}" does not exist in this fieldset'.format(name)
            )
        vkwargs = {}
        if self._vocab_options[name]:
            for argument, option_name in self._vocab_options[name].items():
                option_value = opt.vocabulary_options.get(option_name)
                if option_value is not None:
                    vkwargs[argument] = option_value
        return vkwargs

    def vocab_vectors_loader(
        self,
        name,
        opt,
        embeddings_format='polyglot',
        embeddings_binary=False
    ):
        if name not in self._vocab_vectors:
            raise KeyError(
                'Field named "{}" does not exist in this fieldset'.format(name)
            )

        vectors_fn = None

        option_name = self._vocab_vectors[name]
        if option_name:
            option_value = opt.vocabulary_options.get(option_name)
            if option_value:
                emb_model = AvailableVectors[embeddings_format]
                vectors_fn = partial(
                    emb_model, option_value, binary=embeddings_binary
                )
        return vectors_fn

    def fields_vocab_options(self, opt):
        vocab_options = {}
        for name, field in self.fields.items():
            vocab_options[name] = dict(
                vectors_fn = self.vocab_vectors_loader(name, opt)
            )
            vocab_options[name].update(self.vocab_kwargs(name, opt))
        return vocab_options