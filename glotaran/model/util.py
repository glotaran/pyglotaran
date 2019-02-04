

def wrap_func_as_method(cls, name=None, annotations=None, doc=None):

    def decorator(func):
        if name:
            func.__name__ = name
        if annotations:
            setattr(func, '__annotations__', annotations)
        if doc:
            func.__doc__ = doc
        func.__qualname__ = cls.__qualname__ + '.' + func.__name__
        func.__module__ = cls.__module__

        return func

    return decorator
