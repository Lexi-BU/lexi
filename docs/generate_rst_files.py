import os
import ast

# Path to the source file
source_file = "../lexi/lexi.py"

# Base directory for the docs
docs_base_dir = "rst_files/"

function_names_dict = {
    "array_to_image": "Array to Image",
    "download_files_from_github": "Download Files from GitHub",
    "get_exposure_maps": "Get Exposure Maps",
    "get_lexi_data": "Get LEXI Data",
    "get_lexi_images": "Get LEXI Images",
    "get_sky_backgrounds": "Get Sky Backgrounds",
    "get_spc_prams": "Get Spacecraft Parameters",
    "validate_input": "Validate Input",
    "vignette": "Vignette",
}


# Template for .rst file content
# Function to generate .rst content for a given function name
def generate_rst(function_name):
    if function_name in function_names_dict:
        title = function_names_dict[function_name]
        # Template for .rst file content with dynamic function name and title
        rst_template = f"""
===================
{title} (`lexi.lexi.{function_name}`)
===================

.. py:currentmodule:: lexi.lexi

.. autofunction:: {function_name}
"""
        return rst_template
    else:
        raise ValueError(
            f"Function name '{function_name}' not found in the dictionary."
        )


def extract_functions_from_file(file_path):
    """
    Extract all function names from the Python file.
    """
    with open(file_path, "r") as f:
        tree = ast.parse(f.read())
    return [node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]


def create_rst_files(function_names):
    """
    Create an .rst file for each function.
    """
    for function_name in function_names:
        # Define folder and file paths
        folder_path = os.path.join(docs_base_dir, function_name)
        file_path = os.path.join(folder_path, "index.rst")

        # Create the folder if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)

        rst_template = generate_rst(function_name)
        # Write the .rst file
        with open(file_path, "w") as f:
            f.write(rst_template.format(function_name=function_name))

        print(f"Created: {file_path}")


def main():
    # Extract function names
    function_names = extract_functions_from_file(source_file)
    print(f"Found functions: {function_names}")

    # Generate .rst files
    create_rst_files(function_names)


if __name__ == "__main__":
    main()
