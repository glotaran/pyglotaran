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
    {% for item in attributes %}
        ~{{ item }}
    {%- endfor %}

    {% endif %}
    {% endblock %}

    {% if '__init__' in methods %}
        {% set caught_result = methods.remove('__init__') %}
    {% endif %}


    {% block methods_summary %}
    {% if methods %}

    .. rubric:: Methods Summary

    .. autosummary::
        :toctree: {{ name }}/methods
        :nosignatures:

    {% for item in methods %}
        ~{{ name }}.{{ item }}
    {%- endfor %}

    {% block methods_documentation %}
    {% if methods %}

    .. rubric:: Methods Documentation

    {% endif %}
    {% endblock %}


    {% endif %}
    {% endblock %}
