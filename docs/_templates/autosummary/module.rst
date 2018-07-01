{% extends "known_packages.rst" %}
{% block module %}
{{ fullname | replace("glotaran.", "") | escape | underline}}
{#{% set reduced_name={child_modules} %}#}

.. currentmodule:: {{ module }}

.. automodule:: {{ fullname }}

    {% if fullname in known_modules %}
Submodules
----------

    .. autosummary::
        {% for item in child_modules %}
            {% if item.startswith( fullname + ".") %}
                {{ item }}
            {% endif %}
        {%- endfor %}
    {% endif %}

    {% block functions %}
    {% if functions %}

Functions
---------

    .. rubric:: Summary

    .. autosummary::
        :toctree: {{ fullname | replace("glotaran.", "") | replace(".", "/") }}/functions
        :nosignatures:
        {% for item in functions %}
        {{ item }}
        {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block classes %}
    {% if classes %}

Classes
-------

    .. rubric:: Summary

    .. autosummary::
        :toctree: {{ fullname | replace("glotaran.", "") | replace(".", "/") }}/classes
        :nosignatures:
    {% for item in classes %}
        {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}

    {% block exceptions %}
    {% if exceptions %}

Exceptions
----------

    .. rubric:: Exception Summary

    .. autosummary::
        :toctree: {{ fullname | replace("glotaran.", "") | replace(".", "/") }}/exceptions
    {% for item in exceptions %}
        {{ item }}
    {%- endfor %}
    {% endif %}
    {% endblock %}


{% endblock %}