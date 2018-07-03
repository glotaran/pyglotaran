#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Helper Module to generate the API documentation located at `docs/generate_api_documentation.py`.

The functionality is available by calling ``make api_docs`` on posix system
or ``make.bat api_docs`` on windows.

If you add ``packages``, ``modules``, ``classes``, ``methods``, ``attributes``,
``functions`` or ``exceptions``, you need might need to run ``make clean_all`` on posix system
or ``make.bat clean_all`` on windows to see changes in the documentation.

The generation of the API is done by traversing the main package
`traverse_module` and listing all child modules for autosummary to process
(see `write_api_documentation`, `api_documentation.rst` and
`_templates/api_documentation_template.rst`).

If the child module is also a package all its contained modules will be listed
(see `write_known_packages`, `known_packages.rst`, `_templates/known_packages_template.rst` and
`_templates/autosummary/module.rst`).

To understand how it works in detail the following links might be of help:

* `Sphinx Templating Docs <http://www.sphinx-doc.org/en/master/templating.html>`_

* `Jinja Templating <http://jinja.pocoo.org/docs/2.10/templates/>`_

* `Sphinx autosummary Docs <http://www.sphinx-doc.org/en/master/usage/extensions/autosummary.html?highlight=autosummary%20>`_

* `Sphinx autodoc Docs  <http://www.sphinx-doc.org/en/master/usage/extensions/autodoc.html#module-sphinx.ext.autodoc>`_
"""

import os
import pkgutil
import logging

DOCS_DIR = os.path.abspath(os.path.dirname(__file__))
TEMPLATE_DIR = os.path.join(DOCS_DIR, "_templates")

API_DOCUMENTATION_TEMPLATE_PATH = os.path.join(TEMPLATE_DIR,
                                               "api_documentation_template.rst")
API_DOCUMENTATION_PATH = os.path.join(DOCS_DIR,
                                      "api_documentation.rst")

KNOWN_PACKAGES_TEMPLATE_PATH = os.path.join(TEMPLATE_DIR,
                                            "known_packages_template.rst")

KNOWN_PACKAGES_PATH = os.path.join(TEMPLATE_DIR,
                                   "known_packages.rst")


def api_generation_logger(heading, msg):
    """
    Helper function to pretty log function output for debug purposes

    Parameters
    ----------
    heading: str
        Heading of the section which should be logged
    msg: str
        Message to be logged
    """
    heading_width = 50
    decoration_str = "\n" + "#"*heading_width + "\n"
    heading = "#" + heading.center(heading_width-2) + "#"
    heading = "{decoration_str}{heading}{decoration_str}\n\n".format(decoration_str=decoration_str,
                                                                     heading=heading
                                                                     )
    logging.info(heading+msg+"\n"*2)


def traverse_package(package_path, project_root, child_modules=[], child_packages=[]):
    """
    Generates lists of `child_modules` and `child_packages` import statement strings,
    which are used by `write_api_documentation` and `write_known_packages`.

    Parameters
    ----------

    package_path : str
        Absolute path to the package, which should be traversed.
        The initial value can be generated with: ``package_name.__path__[0]``

    project_root : str
        Absolute path to the project of the package, which should be traversed.
        The initial value can be generated with:  ``os.path.split(package_name.__path__[0])[0]``

    child_modules : list
        This list is for bookkeeping the import statement strings of all child modules,
        while the packages are traversed and populated for each contained module.
        The default value should only be alter if manually adding a module is desired.

    child_packages : list
        This list is for bookkeeping the import statement strings of all child packages,
        while the packages are traversed and populated for each contained packages.
        The default value should only be alter if manually adding a module is desired.

    Returns
    -------
    dict
        Dict with keys "child_modules" and "child_packages"

        child_modules:
            List of the import statement strings of all modules and packages in the
            initial package, defined by the location `package_path` when first called.

        child_packages:
            List of the import statement strings of all packages in the initial package,
            defined by the location `package_path` when first called.

    Notes
    -----
    Using ``list`` for bookkeeping works in python, since lists as arguments are copied by
    reference and not copied by value, as most other variables.
    Due to recursion all returned values, besides the most outer call, will be ignored.
    """
    # taken from https://stackoverflow.com/questions/1707709/list-all-the-modules-that-are-part-of-a-python-package # noqa:
    for importer, modname, ispkg in pkgutil.iter_modules([package_path]):
        # print("Found submodule {} (is a package: {}) imported by {}".format(modname,
        #                                                                     ispkg,
        #                                                                     importer))
        submodule_path = os.path.abspath(os.path.join(package_path, modname))
        import_path = submodule_path.replace(project_root, "").replace(os.sep, ".")[1:]
        child_modules.append(import_path)

        if ispkg:
            child_packages.append(import_path)
            traverse_package(submodule_path, project_root, child_modules, child_packages)

    if os.path.split(package_path)[0] == project_root:
        msg = "\n".join(child_modules)
        api_generation_logger("CHILD_MODULES", msg)
        msg = "\n".join(child_packages)
        api_generation_logger("CHILD_PACKAGES", msg)

    return {"child_modules": child_modules,
            "child_packages": child_packages}


def write_api_documentation(child_modules,
                            api_documentation_template_path,
                            api_documentation_path):
    """
    Writes a list of all modules and packages which should be documented by
    autosummary to the api documentation at `api_documentation_path` file using
    the template at `api_documentation_template_path`.

    Parameters
    ----------

    child_modules: list
        List of the import statement strings of all modules and packages
        to be documented.

    api_documentation_template_path: str
        Path to the template file for the api documentation file.

    api_documentation_path: str
        Path to the api documentation file, which will be included in Sphinx.

    Notes
    -----
    This is needed since autosummary needs to be told which modules to generate
    documentation for.
    """
    concat_str = "\n    "
    with open(api_documentation_template_path) as template:
        template_str = template.read()

    module_str = concat_str.join(child_modules)
    template_str = template_str.format(module_str)
    with open(api_documentation_path, "w") as doc:
        doc.write(template_str)

    api_generation_logger("API_DOCUMENTATION", template_str)


def write_known_packages(child_packages, child_modules,
                         known_packages_template_path,
                         known_packages_path):
    """
    Writes a list of all modules and packages which should be documented by
    autosummary to the known packages file at `known_packages_path` file using
    the template at `known_packages_template_path`.

    Parameters
    ----------

    child_packages: list
        List of the import statement strings of all packages to be documented.

    child_modules: list
        List of the import statement strings of all modules and packages
        to be documented.

    known_packages_template_path: str
        Path to the template file for the known packages file.

    known_packages_path: str
        Path to the known packages template file, which will be extended by the autosummary
        template `module.rst`.

    Notes
    -----

    Extending `module.rst` with known packages template file allows to to differentiate between
    modules and packages, when autosummary generates the documentation.
    """
    child_packages_str = child_packages.__repr__()
    child_modules_str = child_modules.__repr__()
    with open(known_packages_template_path) as template:
        template_str = template.read()

    template_str = template_str.format(child_packages_str=child_packages_str,
                                       child_modules_str=child_modules_str)
    with open(known_packages_path, "w") as doc:
        doc.write(template_str)

    api_generation_logger("KNOWN_PACKAGES", template_str)


if __name__ == "__main__":
    DOCS_DIR = os.path.abspath(os.path.dirname(__file__))
    TEMPLATE_DIR = os.path.join(DOCS_DIR, "_templates")

    API_DOCUMENTATION_TEMPLATE_PATH = os.path.join(TEMPLATE_DIR,
                                                   "api_documentation_template.rst")
    API_DOCUMENTATION_PATH = os.path.join(DOCS_DIR,
                                          "api_documentation.rst")

    KNOWN_PACKAGES_TEMPLATE_PATH = os.path.join(TEMPLATE_DIR,
                                                "known_packages_template.rst")

    KNOWN_PACKAGES_PATH = os.path.join(TEMPLATE_DIR,
                                       "known_packages.rst")

    # uncomment the next line to log funtionoutput for debugging
    # logging.basicConfig(filename='generate_api_documentation.log', level=logging.INFO)
    import glotaran
    PACKAGE_ROOT = glotaran.__path__[0]
    PROJECT_ROOT = os.path.split(PACKAGE_ROOT)[0]
    module_imports = traverse_package(PACKAGE_ROOT, PROJECT_ROOT)
    child_modules = module_imports["child_modules"]
    child_packages = module_imports["child_packages"]

    write_api_documentation(child_modules,
                            API_DOCUMENTATION_TEMPLATE_PATH,
                            API_DOCUMENTATION_PATH)

    write_known_packages(child_packages, child_modules,
                         KNOWN_PACKAGES_TEMPLATE_PATH,
                         KNOWN_PACKAGES_PATH)
