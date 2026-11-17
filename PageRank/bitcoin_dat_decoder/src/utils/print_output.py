from pathlib import Path

def print_summary(selected_paths, block_count, edge_count, output_path):
    summary = {
        "selected_dat_files": len(selected_paths),
        "selected_paths": selected_paths,
        "block_count": block_count,
        "edge_count": edge_count
    }
    write_summary_on_file(output_path, summary)
    print(output_path.read_text())

def write_summary_on_file(path, summary):
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for key, value in summary.items():
        if isinstance(value, list):
            lines.append(f"{key}:")
            for item in value:
                lines.append(f"  - {item}")
        else:
            lines.append(f"{key}: {value}")
    output.write_text("\n".join(lines) + "\n")