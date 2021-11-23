from typing import Union


def __setattr__(self, name: str, value: Union[Tensor, 'Module']):
    def remove_from(*dicts_or_sets):
        for d in dicts_or_sets:
            if name in d:
                if isinstance(d, dict):
                    del d[name]
                else:
                    d.discard(name)

    params = self.__dict__.get('_parameters')
    if isinstance(value, Parameter):
        if params is None:
            raise AttributeError(
                "cannot assign parameters before Module.__init__() call")
        remove_from(self.__dict__, self._buffers, self._modules, self._non_persistent_buffers_set)
        self.register_parameter(name, value)
    elif params is not None and name in params:
        if value is not None:
            raise TypeError("cannot assign '{}' as parameter '{}' "
                            "(torch.nn.Parameter or None expected)"
                            .format(torch.typename(value), name))
        self.register_parameter(name, value)
    else:
        modules = self.__dict__.get('_modules')
        if isinstance(value, Module):
            if modules is None:
                raise AttributeError(
                    "cannot assign module before Module.__init__() call")
            remove_from(self.__dict__, self._parameters, self._buffers, self._non_persistent_buffers_set)
            modules[name] = value
        elif modules is not None and name in modules:
            if value is not None:
                raise TypeError("cannot assign '{}' as child module '{}' "
                                "(torch.nn.Module or None expected)"
                                .format(torch.typename(value), name))
            modules[name] = value
        else:
            buffers = self.__dict__.get('_buffers')
            if buffers is not None and name in buffers:
                if value is not None and not isinstance(value, torch.Tensor):
                    raise TypeError("cannot assign '{}' as buffer '{}' "
                                    "(torch.Tensor or None expected)"
                                    .format(torch.typename(value), name))
                buffers[name] = value
            else:
                object.__setattr__(self, name, value)