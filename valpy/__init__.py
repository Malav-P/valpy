hard_dependencies = (
    "numpy",
    "rudolfpy"
)

missing_dependencies = []
missing_names = []

for dependency in hard_dependencies:
    try:
        __import__(dependency)
    except ImportError as e:
        missing_names.append(dependency)

        if dependency != "rudolfpy":
            missing_dependencies.append(f"{dependency}: {e}")

if "rudolfpy" in missing_names:
    raise ImportError("rudolfy not found. To download, please clone the repo using `git clone -b dev https://github.gatech.edu/SSOG/python-filter.git` and add to PATH.")

if missing_dependencies:
    raise ImportError(
        "Unable to import required dependencies:\n" + "\n".join(missing_dependencies)
    )
del hard_dependencies, dependency, missing_dependencies

from .model import ValidationModel