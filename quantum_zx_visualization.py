"""
Quantum ZX-Diagram SVG Visualization

Generates clean SVG visualizations from ZX-diagram data structures.
Uses standard ZX-calculus color conventions:
- Z-spiders: Green
- X-spiders: Red  
- H-boxes: Yellow

Layer 2 visualization (deterministic, 0 tokens).
"""

from typing import Dict, List, Tuple, Optional
from quantum_zx_calculus import ZXDiagram


# Color scheme for ZX-calculus
SPIDER_COLORS = {
    "Z": "#4caf50",  # Green for Z-spiders
    "X": "#f44336",  # Red for X-spiders
    "H": "#ffd54f",  # Yellow for H-boxes
}

SPIDER_STROKE = {
    "Z": "#2e7d32",  # Dark green
    "X": "#c62828",  # Dark red
    "H": "#f9a825",  # Dark yellow
}


def generate_svg_diagram(
    zx_diagram: ZXDiagram,
    width: int = 800,
    height: int = 400,
    spider_radius: int = 25,
    show_phases: bool = True,
    show_labels: bool = True
) -> str:
    """
    Generate SVG visualization of ZX-diagram.
    
    Args:
        zx_diagram: ZXDiagram object from circuit_to_zx_diagram
        width: SVG canvas width
        height: SVG canvas height
        spider_radius: Radius of spider circles
        show_phases: Display phase values on spiders
        show_labels: Display gate labels on spiders
    
    Returns:
        SVG string ready for display or file output
    """
    
    # Calculate layout
    positions = _calculate_spider_positions(
        zx_diagram, width, height, spider_radius
    )
    
    # Build SVG
    svg_parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'style="font-family: \'SF Mono\', Monaco, monospace; background: white;">',
        
        # Title
        f'<text x="{width/2}" y="30" font-size="16" font-weight="bold" '
        f'text-anchor="middle" fill="#333">ZX-Diagram Visualization</text>',
    ]
    
    # Draw wires first (so they appear behind spiders)
    svg_parts.append(_draw_wires(zx_diagram, positions))
    
    # Draw spiders
    svg_parts.append(_draw_spiders(
        zx_diagram, positions, spider_radius, show_phases, show_labels
    ))
    
    # Add legend
    svg_parts.append(_draw_legend(width, height, spider_radius))
    
    svg_parts.append('</svg>')
    
    return '\n'.join(svg_parts)


def _calculate_spider_positions(
    zx_diagram: ZXDiagram,
    width: int,
    height: int,
    spider_radius: int
) -> Dict[int, Tuple[int, int]]:
    """
    Calculate (x, y) positions for each spider using simple layered layout.
    
    Strategy: Arrange spiders in layers from left to right based on 
    their connectivity, simulating circuit flow.
    """
    positions = {}
    
    num_spiders = len(zx_diagram.spiders)
    if num_spiders == 0:
        return positions
    
    # Simple layering: assign each spider to a layer based on label order
    # (More sophisticated layouts could use topological sort)
    
    margin = 80
    usable_width = width - 2 * margin
    usable_height = height - 2 * margin - 60  # Account for title and legend
    
    # Get bounding box from diagram or use defaults
    if zx_diagram.bounding_box:
        num_qubits, depth = zx_diagram.bounding_box
    else:
        # Estimate from spider count
        num_qubits = max(2, int((num_spiders / 2) ** 0.5))
        depth = max(2, num_spiders // num_qubits)
    
    # Calculate spacing
    h_spacing = usable_width / max(1, depth + 1)
    v_spacing = usable_height / max(1, num_qubits - 1) if num_qubits > 1 else usable_height / 2
    
    # Assign positions
    for spider in zx_diagram.spiders:
        spider_id = spider['id']
        
        # Extract layer from label if possible (e.g., "q0_init" = layer 0)
        label = spider.get('label', '')
        
        # Simple heuristic: spread spiders across width
        layer = spider_id % (depth + 1)
        qubit = spider_id // (depth + 1) if depth > 0 else spider_id % num_qubits
        
        x = margin + layer * h_spacing
        y = margin + 60 + qubit * v_spacing
        
        positions[spider_id] = (int(x), int(y))
    
    return positions


def _draw_wires(
    zx_diagram: ZXDiagram,
    positions: Dict[int, Tuple[int, int]]
) -> str:
    """Draw wires connecting spiders."""
    
    wire_svg = ['<g id="wires">']
    
    for from_id, to_id in zx_diagram.wires:
        if from_id not in positions or to_id not in positions:
            continue
        
        x1, y1 = positions[from_id]
        x2, y2 = positions[to_id]
        
        wire_svg.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" '
            f'stroke="#666" stroke-width="2" />'
        )
    
    wire_svg.append('</g>')
    return '\n'.join(wire_svg)


def _draw_spiders(
    zx_diagram: ZXDiagram,
    positions: Dict[int, Tuple[int, int]],
    spider_radius: int,
    show_phases: bool,
    show_labels: bool
) -> str:
    """Draw spider nodes with colors, phases, and labels."""
    
    spider_svg = ['<g id="spiders">']
    
    for spider in zx_diagram.spiders:
        spider_id = spider['id']
        spider_type = spider['type']
        label = spider.get('label', '')
        
        if spider_id not in positions:
            continue
        
        x, y = positions[spider_id]
        phase = zx_diagram.phases.get(spider_id, 0.0)
        
        # Get colors
        fill_color = SPIDER_COLORS.get(spider_type, "#cccccc")
        stroke_color = SPIDER_STROKE.get(spider_type, "#666666")
        
        # Draw spider circle or box
        if spider_type == "H":
            # H-box is square
            size = spider_radius * 1.5
            spider_svg.append(
                f'<rect x="{x-size}" y="{y-size}" width="{size*2}" height="{size*2}" '
                f'fill="{fill_color}" stroke="{stroke_color}" stroke-width="2" />'
            )
            spider_svg.append(
                f'<text x="{x}" y="{y+6}" font-size="18" font-weight="bold" '
                f'text-anchor="middle" fill="#333">H</text>'
            )
        else:
            # Z and X spiders are circles
            spider_svg.append(
                f'<circle cx="{x}" cy="{y}" r="{spider_radius}" '
                f'fill="{fill_color}" stroke="{stroke_color}" stroke-width="2" />'
            )
            spider_svg.append(
                f'<text x="{x}" y="{y+6}" font-size="16" font-weight="bold" '
                f'text-anchor="middle" fill="white">{spider_type}</text>'
            )
        
        # Show phase if non-zero
        if show_phases and phase != 0.0:
            phase_text = f"{phase}π" if phase != 1.0 else "π"
            spider_svg.append(
                f'<text x="{x}" y="{y-spider_radius-8}" font-size="11" '
                f'text-anchor="middle" fill="#333" font-weight="bold">{phase_text}</text>'
            )
        
        # Show label
        if show_labels and label:
            spider_svg.append(
                f'<text x="{x}" y="{y+spider_radius+18}" font-size="9" '
                f'text-anchor="middle" fill="#666">{label}</text>'
            )
    
    spider_svg.append('</g>')
    return '\n'.join(spider_svg)


def _draw_legend(width: int, height: int, spider_radius: int) -> str:
    """Draw legend explaining spider types."""
    
    legend_y = height - 50
    legend_x_start = 50
    spacing = 150
    
    legend_svg = ['<g id="legend">']
    
    # Z-spider
    x = legend_x_start
    legend_svg.append(
        f'<circle cx="{x}" cy="{legend_y}" r="{spider_radius//2}" '
        f'fill="{SPIDER_COLORS["Z"]}" stroke="{SPIDER_STROKE["Z"]}" stroke-width="1.5" />'
    )
    legend_svg.append(
        f'<text x="{x}" y="{legend_y+4}" font-size="10" font-weight="bold" '
        f'text-anchor="middle" fill="white">Z</text>'
    )
    legend_svg.append(
        f'<text x="{x+25}" y="{legend_y+4}" font-size="10" fill="#333">Z-spider</text>'
    )
    
    # X-spider
    x = legend_x_start + spacing
    legend_svg.append(
        f'<circle cx="{x}" cy="{legend_y}" r="{spider_radius//2}" '
        f'fill="{SPIDER_COLORS["X"]}" stroke="{SPIDER_STROKE["X"]}" stroke-width="1.5" />'
    )
    legend_svg.append(
        f'<text x="{x}" y="{legend_y+4}" font-size="10" font-weight="bold" '
        f'text-anchor="middle" fill="white">X</text>'
    )
    legend_svg.append(
        f'<text x="{x+25}" y="{legend_y+4}" font-size="10" fill="#333">X-spider</text>'
    )
    
    # H-box
    x = legend_x_start + 2 * spacing
    size = spider_radius // 2
    legend_svg.append(
        f'<rect x="{x-size}" y="{legend_y-size}" width="{size*2}" height="{size*2}" '
        f'fill="{SPIDER_COLORS["H"]}" stroke="{SPIDER_STROKE["H"]}" stroke-width="1.5" />'
    )
    legend_svg.append(
        f'<text x="{x}" y="{legend_y+4}" font-size="10" font-weight="bold" '
        f'text-anchor="middle" fill="#333">H</text>'
    )
    legend_svg.append(
        f'<text x="{x+25}" y="{legend_y+4}" font-size="10" fill="#333">Hadamard</text>'
    )
    
    legend_svg.append('</g>')
    return '\n'.join(legend_svg)


def save_svg_to_file(svg_content: str, filename: str) -> None:
    """
    Save SVG content to file.
    
    Args:
        svg_content: SVG string from generate_svg_diagram
        filename: Output filename (e.g., "circuit.svg")
    """
    with open(filename, 'w') as f:
        f.write(svg_content)
    print(f"SVG saved to {filename}")


def display_in_notebook(svg_content: str):
    """
    Display SVG in Jupyter notebook.
    
    Args:
        svg_content: SVG string from generate_svg_diagram
    """
    from IPython.display import SVG, display
    display(SVG(svg_content))


# Example usage
if __name__ == "__main__":
    from quantum_zx_calculus import parse_qasm_circuit, circuit_to_zx_diagram
    
    # Example: Bell state circuit
    bell_qasm = """
    OPENQASM 2.0;
    include "qelib1.inc";
    qreg q[2];
    h q[0];
    cx q[0], q[1];
    """
    
    circuit = parse_qasm_circuit(bell_qasm)
    zx_diagram = circuit_to_zx_diagram(circuit)
    
    svg = generate_svg_diagram(zx_diagram)
    save_svg_to_file(svg, "bell_state_zx.svg")
    
    print("Generated ZX-diagram visualization")
    print(f"Spiders: {len(zx_diagram.spiders)}")
    print(f"Wires: {len(zx_diagram.wires)}")
