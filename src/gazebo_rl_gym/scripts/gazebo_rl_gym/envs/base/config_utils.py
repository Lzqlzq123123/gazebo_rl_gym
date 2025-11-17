import collections.abc


class SimpleNamespace:
    """Simple namespace object to hold attributes dynamically."""
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)


def update_config_from_dict(cfg, overrides):
    """Recursively update a config object using a dictionary.
    
    If a key doesn't exist in cfg and the value is a dict, create a SimpleNamespace for it.
    """
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            # If the value is a dict and the attribute doesn't exist, create a SimpleNamespace
            if isinstance(value, collections.abc.Mapping):
                ns = SimpleNamespace()
                update_config_from_dict(ns, value)
                setattr(cfg, key, ns)
            else:
                setattr(cfg, key, value)
        else:
            attr = getattr(cfg, key)
            if isinstance(value, collections.abc.Mapping) and not isinstance(attr, (int, float, str, bool, list, tuple, set, type(None))):
                update_config_from_dict(attr, value)
            else:
                setattr(cfg, key, value)
