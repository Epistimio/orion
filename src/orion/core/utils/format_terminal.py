"""
Utility functions for formatting prints to terminal
===================================================

Functions to build strings for terminal prints

"""


INFO_TEMPLATE = """\
{identification}

{commandline}

{configuration}

{algorithm}

{space}

{metadata}

{refers}

{stats}
"""


def format_info(experiment):
    """Render a string for all info of experiment"""
    info_string = INFO_TEMPLATE.format(
        identification=format_identification(experiment),
        commandline=format_commandline(experiment),
        configuration=format_config(experiment),
        algorithm=format_algorithm(experiment),
        space=format_space(experiment),
        metadata=format_metadata(experiment),
        refers=format_refers(experiment),
        stats=format_stats(experiment),
    )

    return info_string


TITLE_TEMPLATE = """\
{title}
{empty:=<{title_len}}\
"""


def format_title(title):
    """Render a title above an horizontal bar"""
    title_string = TITLE_TEMPLATE.format(title=title, title_len=len(title), empty="")

    return title_string


DICT_EMPTY_LEAF_TEMPLATE = "{tab}{key}\n"
DICT_LEAF_TEMPLATE = "{tab}{key}: {value}\n"
DICT_NODE_TEMPLATE = "{tab}{key}:\n{value}\n"


def format_dict(dictionary, depth=0, width=4, templates=None):
    r"""Render a dict on multiple lines

    Parameters
    ----------
    dictionary: dict
        The dictionary to render
    depth: int
        Tab added at the beginning of every lines
    width: int
        Size of the tab added to each line, multiplied
        by the depth of the object in the dict of dicts.
    templates: dict
        Templates for `empty_leaf`, `leaf` and `dict_node`.
        Default is
        `empty_leaf="{tab}{key}"`
        `leaf="{tab}{key}: {value}\n"`
        `dict_node="{tab}{key}:\n{value}\n"`

    Examples
    --------
    >>> print(format_dict({1: {2: 3, 3: 4}, 2: {3: 4, 4: {5: 6}}}))
    1:
        2: 3
        3: 4
    2:
        3: 4
        4:
            5: 6
    >>> templates = {'leaf': '{tab}{key}={value}\n', 'dict_node': '{tab}{key}:\n{value}\n'}
    >>> print(format_dict({1: {2: 3, 3: 4}, 2: {3: 4, 4: {5: 6}}}, templates=templates))
    1:
        2=3
        3=4
    2:
        3=4
        4:
            5=6

    """
    if isinstance(dictionary, (list, tuple)):
        return format_list(dictionary, depth, width=width, templates=templates)

    # To avoid using mutable objects as default values in function signature.
    if templates is None:
        templates = {}

    empty_leaf_template = templates.get("empty_leaf", DICT_EMPTY_LEAF_TEMPLATE)
    leaf_template = templates.get("leaf", DICT_LEAF_TEMPLATE)
    node_template = templates.get("dict_node", DICT_NODE_TEMPLATE)

    dict_string = ""
    for key in sorted(dictionary.keys()):
        tab = " " * (depth * width)
        value = dictionary[key]
        if isinstance(value, (dict, list, tuple)):
            if not value:
                dict_string += empty_leaf_template.format(tab=tab, key=key)
            else:
                subdict_string = format_dict(
                    value, depth + 1, width=width, templates=templates
                )
                dict_string += node_template.format(
                    tab=tab, key=key, value=subdict_string
                )
        else:
            dict_string += leaf_template.format(tab=tab, key=key, value=value)

    return dict_string.replace(" \n", "\n").rstrip("\n")


LIST_TEMPLATE = """\
{tab}[
{items}
{tab}]\
"""
LIST_ITEM_TEMPLATE = "{tab}{item}\n"
LIST_NODE_TEMPLATE = "{item}\n"


def format_list(a_list, depth=0, width=4, templates=None):
    r"""Render a list on multiple lines

    Parameters
    ----------
    a_list: list
        The list to render
    depth: int
        Tab added at the beginning of every lines
    width: int
        Size of the tab added to each line, multiplied
        by the depth of the object in the list of lists.
    templates: dict
        Templates for `list`, `item` and `list_node`.
        Default is
        `list="{tab}[\n{items}\n{tab}]"`
        `item="{tab}{item}\n"`
        `list_node="{item}\n"`

    Examples
    --------
    >>> print(format_list([1, [2, 3], 4, [5, 6, 7, 8]]))
    [
        1
        [
            2
            3
        ]
        4
        [
            5
            6
            7
            8
        ]
    ]
    >>> templates = {}
    >>> templates['list'] = '{tab}\n{items}\n{tab}'
    >>> templates['item'] = '{tab}- {item}\n'
    >>> templates['list_node'] = '{tab}{item}\n'
    >>> print(format_list([1, [2, 3], 4, [5, 6, 7, 8]], width=2, templates=templates))
      - 1

        - 2
        - 3

      - 4

        - 5
        - 6
        - 7
        - 8

    """
    # To avoid using mutable objects as default values in function signature.
    if templates is None:
        templates = {}

    list_template = templates.get("list", LIST_TEMPLATE)
    item_template = templates.get("item", LIST_ITEM_TEMPLATE)
    node_template = templates.get("list_node", LIST_NODE_TEMPLATE)

    tab = " " * (depth * width)
    list_string = ""
    for i, item in enumerate(a_list, 1):
        subtab = " " * ((depth + 1) * width)
        if isinstance(item, (dict, list, tuple)):
            item_string = format_dict(item, depth + 1, width=width, templates=templates)
            list_string += node_template.format(tab=subtab, id=i, item=item_string)
        else:
            list_string += item_template.format(tab=subtab, id=i, item=item)

    return list_template.format(tab=tab, items=list_string.rstrip("\n"))


ID_TEMPLATE = """\
{title}
name: {name}
version: {version}
user: {user}
"""


def format_identification(experiment):
    """Render a string for identification section"""
    identification_string = ID_TEMPLATE.format(
        title=format_title("Identification"),
        name=experiment.name,
        version=experiment.version,
        user=experiment.metadata["user"],
    )

    return identification_string


COMMANDLINE_TEMPLATE = """\
{title}
{commandline}
"""


def format_commandline(experiment):
    """Render a string for commandline section"""
    if "user_args" not in experiment.metadata:
        return ""

    commandline_string = COMMANDLINE_TEMPLATE.format(
        title=format_title("Commandline"),
        commandline=" ".join(experiment.metadata["user_args"]),
    )

    return commandline_string


CONFIG_TEMPLATE = """\
{title}
max trials: {experiment.max_trials}
max broken: {experiment.max_broken}
working dir: {experiment.working_dir}
"""


def format_config(experiment):
    """Render a string for config section"""
    config_string = CONFIG_TEMPLATE.format(
        title=format_title("Config"), experiment=experiment
    )

    return config_string


ALGORITHM_TEMPLATE = """\
{title}
{configuration}
"""


def format_algorithm(experiment):
    """Render a string for algorithm section"""
    algorithm_string = ALGORITHM_TEMPLATE.format(
        title=format_title("Algorithm"),
        configuration=format_dict(experiment.configuration["algorithms"]),
    )

    return algorithm_string


SPACE_TEMPLATE = """\
{title}
{params}
"""


def format_space(experiment):
    """Render a string for space section"""
    space_string = SPACE_TEMPLATE.format(
        title=format_title("Space"),
        params="\n".join(
            name + ": " + experiment.space[name].get_prior_string()
            for name in experiment.space.keys()
        ),
    )

    return space_string


METADATA_TEMPLATE = """\
{title}
user: {experiment.metadata[user]}
datetime: {experiment.metadata[datetime]}
orion version: {experiment.metadata[orion_version]}
VCS:
{vcs}
"""


def format_metadata(experiment):
    """Render a string for metadata section"""
    metadata_string = METADATA_TEMPLATE.format(
        title=format_title("Meta-data"),
        experiment=experiment,
        vcs=format_dict(experiment.metadata.get("VCS", {}), depth=1, width=2),
    )

    return metadata_string


REFERS_TEMPLATE = """\
{title}
root:{root}
parent:{parent}
adapter:{adapter}
"""


def format_refers(experiment):
    """Render a string for refers section"""
    if experiment.node.root is experiment.node:
        root = ""
        parent = ""
        adapter = ""
    else:
        root = experiment.node.root.name
        parent = experiment.node.parent.name
        adapter = "\n" + format_dict(
            experiment.refers["adapter"].configuration, depth=1, width=2
        )

    refers_string = REFERS_TEMPLATE.format(
        title=format_title("Parent experiment"),
        root=(" " + root) if root else "",
        parent=(" " + parent) if parent else "",
        adapter=adapter,
    )

    return refers_string


STATS_TEMPLATE = """\
{title}
completed: {is_done}
trials completed: {stats.trials_completed}
best trial:
  id: {stats.best_trials_id}
  evaluation: {stats.best_evaluation}
  params:
{best_params}
start time: {stats.start_time}
finish time: {stats.finish_time}
duration: {stats.duration}
"""


NO_STATS_TEMPLATE = """\
{title}
No trials executed...
"""


def format_stats(experiment):
    """Render a string for stat section

    Parameters
    ----------
    experiment: `orion.core.worker.experiment.Experiment`
    templates: dict
        templates for the title and `stats`.
        See `format_title` for more info.

    """
    stats = experiment.stats
    if not stats:
        return NO_STATS_TEMPLATE.format(title=format_title("Stats"))

    best_params = get_trial_params(stats.best_trials_id, experiment)

    stats_string = STATS_TEMPLATE.format(
        title=format_title("Stats"),
        stats=stats,
        best_params=format_dict(best_params, depth=2, width=2),
        is_done=experiment.is_done,
    )

    return stats_string


def get_trial_params(trial_id, experiment):
    """Get params from trial_id in given experiment"""
    best_trial = experiment.get_trial(uid=trial_id)
    if not best_trial:
        return {}

    return best_trial.params
