import inspect


class BaseConfig:
    def __init__(self) -> None:
        """Recursively instantiate all nested config classes."""
        self._init_member_classes(self)

    @classmethod
    def _init_member_classes(cls, obj):
        for name in dir(obj):
            if name == "__class__":
                continue
            value = getattr(obj, name)
            if inspect.isclass(value):
                instance = value()
                setattr(obj, name, instance)
                cls._init_member_classes(instance)
