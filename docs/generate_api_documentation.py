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


def traverse_module(module_path, project_root, child_modules=[], child_packages=[]):
    for importer, modname, ispkg in pkgutil.iter_modules([module_path]):
        # print("Found submodule %s (is a package: %s) " % (modname, ispkg))
        submodule_path = os.path.abspath(os.path.join(module_path, modname))
        import_path = submodule_path.replace(project_root, "").replace(os.sep, ".")[1:]
        child_modules.append(import_path)

        if ispkg:
            child_packages.append(str(import_path))
            traverse_module(submodule_path, project_root, child_modules, child_packages)
    return {"child_modules": child_modules,
            "child_packages": child_packages}


def write_api_documentation(child_modules):
    concat_str = "\n    "
    with open(API_DOCUMENTATION_TEMPLATE_PATH) as template:
        template_str = template.read()

    module_str = concat_str.join(child_modules)
    template_str = template_str.format(module_str)
    with open(API_DOCUMENTATION_PATH, "w") as doc:
        doc.write(template_str)


def write_known_packages(child_packages, child_modules):
    child_packages_str = child_packages.__repr__()
    child_modules_str = child_modules.__repr__()
    with open(KNOWN_PACKAGES_TEMPLATE_PATH) as template:
        template_str = template.read()

    template_str = template_str.format(child_packages_str=child_packages_str,
                                       child_modules_str=child_modules_str)
    with open(KNOWN_PACKAGES_PATH, "w") as doc:
        doc.write(template_str)


if __name__ == "__main__":
    print_sep = "\n" + "#"*30 + "\n"
    import glotaran
    PACKAGE_ROOT = glotaran.__path__[0]
    PROJECT_ROOT = os.path.split(PACKAGE_ROOT)[0]
    module_imports = traverse_module(PACKAGE_ROOT, PROJECT_ROOT)
    child_modules = module_imports["child_modules"]
    child_packages = module_imports["child_packages"]
    #
    # print(print_sep, "CHILD MODULES:", print_sep)
    # for submodule in child_modules:
    #     print(submodule)
    #
    # print(print_sep, "CHILD PACKAGES:", print_sep)
    # for submodule in child_packages:
    #     print(submodule)

    # write_api_documentation(child_modules)
    write_known_packages(child_packages, child_modules)
