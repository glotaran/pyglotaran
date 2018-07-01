import os
import pkgutil

DOCS_DIR = os.path.abspath(os.path.dirname(__file__))
API_PAGES_DIR = os.path.join(DOCS_DIR, "api_pages")

if not os.path.isdir(API_PAGES_DIR):
    os.makedirs(API_PAGES_DIR)

PACKAGE_RST="""

"""

MODULE_RST= """
.. currentmodule:: {parent_package}

.. autosummary::
   :toctree: api/{rel_package_path}

   {module_name}
"""

def traverse_module(base_module_path, abs_root_dir):
    child_modules = []
    has_child_packages = False
    for importer, modname, ispkg in pkgutil.iter_modules([base_module_path]):
        print("Found submodule %s (is a package: %s)" % (modname, ispkg))
        module_path = os.path.join(base_module_path, modname)
        rel_module_path = os.path.relpath(module_path, start=abs_root_dir)
        parent_module_name = os.path.split(importer.path)[1]
        rst_path = os.path.join(parent_module_name, parent_module_name + ".rst")
        child_modules.append(modname)

        if ispkg:
            has_child_packages = True
            # if not os.path.isfile(rst_path):
            #     with open(rst_path, "w", encoding="utf8") as rst_file:
            #         rst_file.write()
            api_pages_subdir = os.path.join(API_PAGES_DIR, rel_module_path)
            if not os.path.isdir(api_pages_subdir):
                os.makedirs(api_pages_subdir)
            traverse_module(module_path, abs_root_dir)
    print(child_modules)


if __name__ == "__main__":
    import glotaran
    ROOT_DIR = glotaran.__path__[0]
    traverse_module(ROOT_DIR, ROOT_DIR)
