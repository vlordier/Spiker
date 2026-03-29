import subprocess as sp
from pathlib import Path


def write_file(component, output_dir: str = "output", rm: bool = False) -> None:
    output_path = Path(output_dir)

    if hasattr(component, "entity"):
        output_file_name = output_path / f"{component.entity.name}.vhd"

    elif hasattr(component, "name"):
        output_path = output_path / f"{component.name}.vhd"

    else:
        raise TypeError("Component is not a valid VHDL object")

    if hasattr(component, "code") and callable(component.code):
        hdl_code = component.code()
    else:
        raise TypeError("Component has no code method")

    if rm:
        sp.run(f"rm -rf {output_dir}", shell=True)

    if not output_path.parent.is_dir():
        sp.run(f"mkdir {output_dir}", shell=True)

    with open(output_file_name, "w+") as fp:
        for line in hdl_code:
            fp.write(line)
