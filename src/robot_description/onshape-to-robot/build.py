import os
import sys
import shutil
import json
import xml.etree.ElementTree as ET
import onshape_to_robot_monkeypatch
from onshape_to_robot import export


def load_config(config_path: str) -> dict:
    with open(config_path, "r") as f:
        return json.load(f)


def export_robot(config_dir: str):
    original_argv = sys.argv.copy()
    sys.argv = ["onshape-to-robot", config_dir]
    try:
        export.main()
    finally:
        sys.argv = original_argv


def post_import_commands(
    base_dir: str, output_filename: str, robot_name: str, assets_directory: str
):
    urdf_output_dir = os.path.join(base_dir, "../../../urdf")
    os.makedirs(urdf_output_dir, exist_ok=True)

    shutil.copy(
        os.path.join(base_dir, f"{output_filename}.urdf"),
        os.path.join(urdf_output_dir, f"{robot_name}.urdf"),
    )

    meshes_src = os.path.join(base_dir, f"{assets_directory}/merged")
    meshes_dst = os.path.join(base_dir, "../../../meshes")

    if os.path.exists(meshes_src):
        shutil.copytree(meshes_src, meshes_dst, dirs_exist_ok=True)


def clean_base_link(urdf_path: str):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for i, link in enumerate(root.findall("link")):
        if link.attrib.get("name") == "base_link":
            empty_base = ET.Element("link", {"name": "base_link"})
            empty_base.tail = "\n"
            root.remove(link)
            root.insert(i, empty_base)
            break

    tree.write(urdf_path)


def replace_meshes_path(urdf_path: str, assets_directory: str):
    tree = ET.parse(urdf_path)
    root = tree.getroot()

    for mesh in root.iter("mesh"):
        filename = mesh.attrib.get("filename")
        if filename and filename.startswith(f"package://{assets_directory}/merged"):
            new_filename = filename.replace(
                f"package://{assets_directory}/merged",
                "package://robot_description/meshes",
            )
            mesh.set("filename", new_filename)

    tree.write(urdf_path)


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(current_dir, "models")

    for model_name in os.listdir(models_dir):
        config_dir = os.path.join(models_dir, model_name)

        if not os.path.isdir(config_dir):
            continue

        config_path = os.path.join(config_dir, "config.json")

        if not os.path.exists(config_path):
            continue

        config = load_config(config_path)

        assets_directory = config.get("assets_directory") or "assets"
        output_filename = (
            config["output_filename"] if "output_filename" in config else "robot"
        )
        robot_name = config.get("robot_name") or os.path.basename(config_dir)

        export_robot(config_dir)

        urdf_path = os.path.join(config_dir, f"{output_filename}.urdf")

        clean_base_link(urdf_path)
        replace_meshes_path(urdf_path, assets_directory)
        post_import_commands(
            config_dir,
            output_filename,
            robot_name,
            assets_directory,
        )
