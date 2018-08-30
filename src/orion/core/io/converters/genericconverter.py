from collections import (defaultdict, deque)
import importlib

from orion.core.io.converters.base import BaseConverter
from orion.core.utils import nesteddict


class GenericConverter(BaseConverter):
    """Generic converter for any configuration file type.

    For each parameter dimension declared here, one must necessarily
    provide a ``name`` keyword inside the `Dimension` building expression.

    Implementation details: As this class is supposed to provide with a
    generic text parser, semantics are going to be tied to their consequent
    usage. A template document is going to be created on `parse` and filled
    with values on `read`. This template document consists the state of this
    `Converter` object.

    Dimension should be defined for instance as:
    ``meaningful_name~uniform(0, 4)``

    """

    def __init__(self, regex=None, expression_prefix=''):
        """Initialize with the regex expression which will be searched for
        to define a `Dimension`.
        """
        self.re_module = importlib.import_module('re')

        if regex is not None:
            self.regex = self.re_module.compile(regex)
        else:
            self.regex = self.re_module.compile(r'([\/]?[\w|\/|-]+)~([\+]?.*\)|\-|\>[A-Za-z_]\w*)')
        self.expression_prefix = expression_prefix
        self.template = None
        self.has_leading = defaultdict(str)
        self.conflict_msg = "Namespace conflict in configuration file '{}', under '{}'"

    def _raise_conflict(self, path, namespace):
        raise ValueError(self.conflict_msg.format(path, namespace))

    def parse(self, filepath):
        r"""Read dictionary out of the configuration file.

        Create a template for Python 3 string format and save it as this
        object's state, by substituing '{\1}' wherever the pattern
        was matched. By default, the first matched group (\1) corresponds
        with a dimension's namespace.

        .. note:: Namespace in substitution templates does not contain the first '/'.

        Parameters
        ----------
        filepath : str
           Full path to the original user script's configuration.

        """
        with open(filepath) as f:
            self.template = f.read()

        # Search for Oríon semantic pattern
        pairs = self.regex.findall(self.template)
        ret = dict(pairs)

        # Every namespace given should be unique,
        # raise conflict if there are duplicates
        if len(pairs) != len(ret):
            namespaces = list(zip(*pairs))[0]
            for name in namespaces:
                if namespaces.count(name) != 1:
                    self._raise_conflict(filepath, name)

        # Create template using each namespace as format key,
        # exactly as provided by the user
        subst = self.re_module.sub(r'{', r'{{', self.template)
        subst = self.re_module.sub(r'}', r'}}', subst)
        substituted, num_subs = self.regex.subn(r'{\1!s}', subst)
        assert len(ret) == num_subs, "This means an error in the regex. Report bug. Details::\n"\
            "original: {}\n, regex:{}".format(self.template, self.regex)
        self.template = substituted

        # Wrap it in style of what the rest of `Converter`s return
        ret_nested = nesteddict()
        for namespace, expression in ret.items():
            keys = namespace.split('/')
            if not keys[0]:  # It means that user wrote a namespace starting from '/'
                keys = keys[1:]  # Safe because of the regex pattern
                self.has_leading[namespace[1:]] = '/'

            stuff = ret_nested
            for i, key in enumerate(keys[:-1]):
                stuff = stuff[key]
                if isinstance(stuff, str):
                    # If `stuff` is not a dictionary while traversing the
                    # namespace path, then this amounts to a conflict which was
                    # not sufficiently get caught
                    self._raise_conflict(filepath, '/'.join(keys[:i + 1]))
            # If final value is already filled,
            # then this must be also due to a conflict
            if stuff[keys[-1]]:
                self._raise_conflict(filepath, namespace)

            # Keep compatibility with `SpaceBuilder._build_from_config`
            stuff[keys[-1]] = self.expression_prefix + expression

        return ret_nested

    def generate(self, filepath, data):
        """Create a configuration file at `filepath` using dictionary `data`."""
        unnested_data = dict()
        stack = deque()
        stack.append(([], data))
        while True:
            try:
                namespace, stuff = stack.pop()
            except IndexError:
                break
            if isinstance(stuff, dict):
                for k, v in stuff.items():
                    stack.append((['/'.join(namespace + [str(k)])], v))
            else:
                name = namespace[0]
                unnested_data[self.has_leading[name] + name] = stuff

        document = self.template.format(**unnested_data)

        with open(filepath, 'w') as f:
            f.write(document)
