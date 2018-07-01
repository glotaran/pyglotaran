{{ objname | escape | underline}}

.. currentmodule:: {{ module }}

.. automodule:: {{ fullname }}

   {% block functions %}
   {% if functions %}

Functions
---------

   .. rubric:: Summary

   .. autosummary::
      :toctree: {{ name }}/functions
      :nosignatures:
      {% for item in functions %}
         ~{{ item }}
      {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block classes %}
   {% if classes %}

Classes
-------

   .. rubric:: Summary

   .. autosummary::
      :toctree: {{ name }}/classes
      :nosignatures:
   {% for item in classes %}
      ~{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}

   {% block exceptions %}
   {% if exceptions %}
   .. rubric:: Exception Summary

   .. autosummary::
   {% for item in exceptions %}
      ~{{ item }}
   {%- endfor %}
   {% endif %}
   {% endblock %}
