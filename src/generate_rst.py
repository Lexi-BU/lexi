import inspect
from lexi_bu import lexi  # Import your package/module

# Get a list of all functions in lexi module
function_names = [
    name for name, obj in inspect.getmembers(lexi) if inspect.isfunction(obj)
]

# Write the function names to a .rst file
with open("functions.rst", "w") as f:
    f.write(".. autosummary::\n")
    f.write("   :toctree: _autosummary\n")
    f.write("   :template: function.rst\n")
    f.write("   :nosignatures:\n\n")
    for name in function_names:
        f.write(f"   lexi_bu.{name}\n")

# Print the function names
print(function_names)
