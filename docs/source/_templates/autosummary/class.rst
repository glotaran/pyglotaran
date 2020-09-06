{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}
    :show-inheritance:
    :members:
    :undoc-members:
    :inherited-members:

    {% block attributes_summary %}
    {% if attributes %}

    .. rubric:: Attributes Summary

    .. autosummary::
       :toctree:
       :template: autosummary/attribute.rst
    {% for item in attributes %}
        ~{{name}}.{{ item }}
    {%- endfor %}

    {% endif %}
    {% endblock %}


    {% block methods_summary %}
    {% if methods %}

    .. rubric:: Methods Summary

    .. autosummary::
        :toctree:
        :nosignatures:

    {% for item in methods %}
      {% if not item.endswith("__init__") %}
        ~{{ name }}.{{ item }}
      {% endif %}
    {%- endfor %}


    {% endif %}
    {% endblock %}

    {% block methods_documentation %}
    {% if methods %}

    .. rubric:: Methods Documentation

    {% endif %}
    {% endblock %}
