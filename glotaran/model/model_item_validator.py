"""This package contains glotarans model item validator."""


class Validator:
    def __init__(self, model_item):
        self._attributes = getattr(model_item, '_glotaran_attributes')

    def val_model(self, model_item, model, errors=[]):
        for attr, opts in self._attributes.items():
            item = getattr(model_item, attr)
            if item is None:
                continue
            if 'target' in opts:
                val_model_opts(item, model, opts, errors)
            elif hasattr(model, attr):
                val_model_attr(item, model, attr, errors)
            else:
                val_model_nested(item, model, errors)

        return errors

    def val_parameter(self, model_item, model, parameter, errors=[]):
        for attr, opts in self._attributes.items():
            item = getattr(model_item, attr)
            if item is None:
                continue
            if 'target' in opts:
                val_parameter_opts(item, parameter, opts, errors)
            elif hasattr(model, attr):
                continue
            else:
                val_parameter_nested(item, model, parameter, errors)

        return errors


def val_model_opts(item, model, opts, errors):
    target = opts['target']
    if isinstance(target, tuple):
        (k_check, v_check) = target
        for k, v in item.items():
            if not k_check == 'parameter':
                model_attr = getattr(model, k_check)
                if not isinstance(k, (list, tuple, set)):
                    k = [k]
                for i in k:
                    if i not in model_attr:
                        errors.append(f"Missing '{k_check}' with label '{i}'")
            if not v_check == 'parameter':
                model_attr = getattr(model, v_check)
                if not isinstance(v, (list, tuple, set)):
                    v = [v]
                for i in v:
                    if i not in model_attr:
                        errors.append(f"Missing '{v_check}' with label '{i}'")
    elif not target == 'parameter':
        model_attr = getattr(model, target)
        if item not in model_attr:
            errors.append(f"Missing '{target}' with label '{item}'")


def val_model_attr(labels, model, attr, errors):
    if not isinstance(labels, list):
        labels = [labels]
    model_attr = getattr(model, attr)
    for label in labels:
        if label not in model_attr and label is not None:
            errors.append(f"Missing '{attr}' with label '{label}'")


def val_model_nested(nested, model, errors):
    if not isinstance(nested, list):
        nested = [nested]
    for n in nested:
        if hasattr(n, "_glotaran_model_item"):
            n.validate_model(model, errors=errors)


def val_parameter_opts(item, parameter, opts, errors):
    target = opts['target']
    if isinstance(target, tuple):
        (k_check, v_check) = target
        for k, v in item.items():
            if k_check == 'parameter':
                if not isinstance(k, (list, tuple, set)):
                    k = [k]
                for i in k:
                    if not parameter.has(i):
                        errors.append(f"Missing parameter with label '{i}'")
            if v_check == 'parameter':
                if not isinstance(v, (list, tuple, set)):
                    v = [v]
                for i in v:
                    if not parameter.has(i):
                        errors.append(f"Missing parameter with label '{i}'")
    elif target == 'parameter':
        if not parameter.has(item):
            errors.append(f"Missing parameter with label '{item}'")


def val_parameter(item, parameter, errors):
    if not isinstance(item, list):
        item = [item]
    if any([not isinstance(i, str) for i in item]):
        return
    for label in item:
        if not parameter.has(label):
            errors.append(f"Missing parameter with label '{label}'")


def val_parameter_nested(nested, model, parameter, errors):
    if not isinstance(nested, list):
        nested = [nested]
    for item in nested:
        if hasattr(item, "_glotaran_model_item"):
            item.validate_parameter(model, parameter, errors=errors)
        else:
            val_parameter(item, parameter, errors)
