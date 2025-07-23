import os
import sys
import shutil
import subprocess
from onshape_to_robot import export
import xml.etree.ElementTree as ET

def export_robot(config_dir: str):
    original_argv = sys.argv.copy()
    sys.argv = ["onshape-to-robot", config_dir]
    try:
        export.main()
    finally:
        sys.argv = original_argv

def post_import_commands(base_dir: str):
    urdf_output_dir = os.path.join(base_dir, "../urdf")
    os.makedirs(urdf_output_dir, exist_ok=True)

    shutil.copy(os.path.join(base_dir, ".urdf"), os.path.join(urdf_output_dir, "r6bot.urdf"))

    meshes_src = os.path.join(base_dir, ".assets/merged")
    meshes_dst = os.path.join(base_dir, "../meshes")
    if os.path.exists(meshes_src):
        if os.path.exists(meshes_dst):
            shutil.rmtree(meshes_dst)
        shutil.copytree(meshes_src, meshes_dst)

def clean_base_link(urdf_path: str):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for link in root.findall("link"):
        if link.attrib.get("name") == "base_link":
            root.remove(link)
            empty_base = ET.Element("link", {"name": "base_link"})
            root.append(empty_base)
            break

    tree.write(urdf_path)
    tree.write(urdf_path)

def replace_meshes_path(urdf_path: str):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for mesh in root.iter('mesh'):
        filename = mesh.attrib.get('filename')
        if filename and filename.startswith('package://.assets/merged'):
            new_filename = filename.replace('package://.assets/merged', 'package://r6bot_description/meshes')
            mesh.set('filename', new_filename)

    tree.write(urdf_path)

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    export_robot(current_dir)

    urdf_path = os.path.join(current_dir, ".urdf")
    clean_base_link(urdf_path)
    replace_meshes_path(urdf_path)

    post_import_commands(current_dir)
