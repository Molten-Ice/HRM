import os
import json  # For handling .ipynb if needed, but since raw text, maybe not necessary

def build_tree(paths):
    tree = {}
    for path in paths:
        parts = path.split(os.sep)
        current = tree
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        current[parts[-1]] = None  # Mark as file
    return tree

def generate_tree_lines(tree, prefix=''):
    lines = []
    items = sorted(tree.keys())
    for i, item in enumerate(items):
        is_last = i == len(items) - 1
        connector = '└── ' if is_last else '├── '
        lines.append(prefix + connector + item)
        if isinstance(tree[item], dict):
            new_prefix = prefix + ('    ' if is_last else '│   ')
            lines.extend(generate_tree_lines(tree[item], new_prefix))
    return lines

# Define image extensions to exclude
image_extensions = {'.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff', '.ico', '.webp', '.svg'}  # Including .svg as image

# Collect all files recursively from the root, excluding images and .git directory
all_files = []
for rootdir, dirs, files in os.walk('.'):
    # Skip .git directory
    if '.git' in dirs:
        dirs.remove('.git')
    for f in files:
        if os.path.splitext(f)[1].lower() in image_extensions:
            continue  # Skip images
        full_path = os.path.join(rootdir, f).lstrip('./')  # Normalize path
        all_files.append(full_path)

# Sort all paths
all_paths = sorted(all_files)

# Build the tree
file_tree = build_tree(all_paths)

# Generate tree diagram lines
tree_diagram = '\n'.join(generate_tree_lines(file_tree))

# Output file
output_file = 'combined.txt'

with open(output_file, 'w', encoding='utf-8') as out_f:
    # Write file hierarchy at the top
    out_f.write("File Structure Diagram:\n")
    out_f.write(tree_diagram + '\n\n')

    # Combine files
    for path in all_paths:
        try:
            with open(path, 'r', encoding='utf-8') as in_f:
                content = in_f.read()
            out_f.write('#' * 10 + ' ' + path + ' ' + '#' * 10 + '\n')
            if path.lower().endswith('.ipynb'):
                # For .ipynb, we already have raw text, but if needed, could json.dump, but raw is fine
                out_f.write(content + '\n\n')
            else:
                out_f.write(content + '\n\n')
        except UnicodeDecodeError:
            # Skip binary files or handle as needed; for now, note it
            out_f.write('#' * 10 + ' ' + path + ' ' + '#' * 10 + '\n')
            out_f.write("[Binary file or encoding issue; content skipped]\n\n")
        except Exception as e:
            out_f.write('#' * 10 + ' ' + path + ' ' + '#' * 10 + '\n')
            out_f.write(f"[Error reading file: {str(e)}]\n\n")

print(f"Output written to {output_file}")