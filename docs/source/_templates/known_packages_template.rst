..
    Don't change known_packages.rst since it changes will be overwritten.
    If you want to change known_packages.rst you have to make the changes in
    known_packages_template.rst and run `make api_docs` afterwards.
    For changes to take effect you might also have to run `make clean_all`
    afterwards.

{{% set known_packages={child_packages_str} %}}
{{% set child_modules={child_modules_str} %}}
{{% block module %}}

{{% endblock %}}
