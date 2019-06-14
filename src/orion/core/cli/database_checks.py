# -*- coding: utf-8 -*-
import yaml


def check_if_config_present(shared_dict):
    """Check if --config is present... """
    if 'config' in shared_dict and shared_dict['config'] is not None:
        return False, ""
    else:
        return True, "The `--config` argument is missing."


def check_if_config_has_database_section(shared_dict):
    """Check if the database section is present... """
    shared_dict = yaml.load(shared_dict['config'])

    if 'database' in shared_dict and shared_dict['database'] is not None:
        return False, ""
    else:
        return True, "No database section inside config file."
