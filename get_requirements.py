import toml

# Read the pyproject.toml file
with open("pyproject.toml", "r") as f:
    pyproject = toml.load(f)

# Extract dependencies from pyproject.toml
dependencies = pyproject.get("tool", {}).get("poetry", {}).get("dependencies", {})
dependency_lines = []

for package, version in dependencies.items():
    if package.lower() == "python":
        continue  # Skip Python version specification
    if isinstance(version, dict):
        # Handle optional dependencies or version constraints
        version_str = version.get("version", "")
        markers = version.get("markers", "")
        dep_line = f"{package}{version_str}"
        if markers:
            dep_line += f" ; {markers}"
        dependency_lines.append(dep_line)
    else:
        dependency_lines.append(f"{package}{version}")

# Write dependencies to requirements.txt
with open("requirements.txt", "w") as req_file:
    req_file.write("\n".join(dependency_lines))

print("requirements.txt has been generated.")
