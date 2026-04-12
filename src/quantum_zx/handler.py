"""
FastMCP handler for Quantum ZX-Calculus server (Categorical Framework).

This server uses the refactored categorical framework to provide:
- Quantum circuit analysis with formal flow verification
- Automatic optimization using ZX-calculus rewrite rules
- Cross-domain transformation capabilities

For FastMCP Cloud deployment, the entry point function must RETURN the server object.
For Claude Desktop, the mcp instance is imported directly.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
from fastmcp import FastMCP

# Create FastMCP instance
mcp = FastMCP("Quantum ZX-Calculus (Categorical Framework)")

# Import categorical framework core (ABSOLUTE IMPORTS - not relative)
from categorical_framework_core import (
    CategoricalDomain, 
    analyze_composition,
    compute_flow_structure,
    RewriteEngine
)

# Import refactored quantum ZX modules (ABSOLUTE IMPORTS)
from quantum_zx_categorical import (
    create_quantum_domain,
    parse_qasm_circuit,
    circuit_to_composition_graph,
    analyze_quantum_circuit
)

# Import domain utilities (ABSOLUTE IMPORTS)
from quantum_zx_domain import (
    lookup_gate,
    get_domain_statistics,
    get_gate_spider_type,
    get_clifford_gates,
    get_t_gates
)

# Initialize quantum domain once at module level (0 tokens)
QUANTUM_DOMAIN = create_quantum_domain()


# ============================================================================
# PHASE 2.6: NORMALIZED 5D PARAMETER SPACE (Morphospace)
# ============================================================================
#
# Maps quantum ZX-calculus visual/structural properties to a continuous
# 5D space [0.0, 1.0]^5 for cross-domain composition with other Lushy
# aesthetic domains.
#
# Dimensions:
#   spider_density      - 0.0 = minimal/identity wires → 1.0 = dense lattice
#   phase_complexity    - 0.0 = Clifford-only (0,π/2,π,3π/2) → 1.0 = arbitrary
#   entanglement_depth  - 0.0 = separable/product state → 1.0 = volume-law
#   chromatic_balance   - 0.0 = Z-spider dominated (green) → 1.0 = X-dominated (red)
#   flow_determinism    - 0.0 = non-deterministic/no flow → 1.0 = fully causal
# ============================================================================

PARAMETER_NAMES: List[str] = [
    "spider_density",
    "phase_complexity",
    "entanglement_depth",
    "chromatic_balance",
    "flow_determinism",
]

# Canonical states: archetypal quantum diagram configurations
CANONICAL_STATES: Dict[str, Dict[str, float]] = {
    "identity_wire": {
        "spider_density": 0.05,
        "phase_complexity": 0.0,
        "entanglement_depth": 0.0,
        "chromatic_balance": 0.50,
        "flow_determinism": 1.0,
    },
    "clifford_stabilizer": {
        "spider_density": 0.45,
        "phase_complexity": 0.15,
        "entanglement_depth": 0.40,
        "chromatic_balance": 0.35,
        "flow_determinism": 0.85,
    },
    "t_magic_state": {
        "spider_density": 0.55,
        "phase_complexity": 0.90,
        "entanglement_depth": 0.50,
        "chromatic_balance": 0.30,
        "flow_determinism": 0.80,
    },
    "bell_entangler": {
        "spider_density": 0.25,
        "phase_complexity": 0.10,
        "entanglement_depth": 0.95,
        "chromatic_balance": 0.50,
        "flow_determinism": 0.90,
    },
    "surface_code_patch": {
        "spider_density": 0.85,
        "phase_complexity": 0.15,
        "entanglement_depth": 0.90,
        "chromatic_balance": 0.50,
        "flow_determinism": 0.95,
    },
    "random_supremacy": {
        "spider_density": 0.90,
        "phase_complexity": 0.85,
        "entanglement_depth": 0.95,
        "chromatic_balance": 0.50,
        "flow_determinism": 0.15,
    },
    "measurement_pattern": {
        "spider_density": 0.60,
        "phase_complexity": 0.45,
        "entanglement_depth": 0.75,
        "chromatic_balance": 0.80,
        "flow_determinism": 0.55,
    },
}


# ============================================================================
# PHASE 2.6: RHYTHMIC PRESETS
# ============================================================================
#
# Period strategy for Tier 4D cross-domain integration:
#   14 — gap-filler (12–15), LCM(14,30)=210, novel harmonics
#   18 — 4-domain overlap (nuclear+catastrophe+diatom+quantum)
#   22 — 3-domain overlap (catastrophe+heraldic+quantum)
#   26 — gap-filler (25–30), LCM(26,30)=390, novel harmonics
#   30 — major hub (microscopy+diatom+heraldic+quantum, 4-domain lock)
# ============================================================================

PHASE26_PRESETS: Dict[str, Dict] = {
    "optimization_cycle": {
        "state_a": "identity_wire",
        "state_b": "clifford_stabilizer",
        "pattern": "sinusoidal",
        "num_cycles": 4,
        "steps_per_cycle": 14,
        "description": (
            "Smooth oscillation between trivial identity and Clifford "
            "stabilizer regime. Models iterative circuit simplification "
            "where complexity builds then collapses through rewriting."
        ),
    },
    "universality_sweep": {
        "state_a": "clifford_stabilizer",
        "state_b": "t_magic_state",
        "pattern": "triangular",
        "num_cycles": 3,
        "steps_per_cycle": 22,
        "description": (
            "Linear ramp between Clifford-only and T-gate-rich regimes. "
            "Traverses the universality boundary where classical "
            "simulability gives way to quantum computational advantage."
        ),
    },
    "entanglement_pulse": {
        "state_a": "identity_wire",
        "state_b": "bell_entangler",
        "pattern": "sinusoidal",
        "num_cycles": 4,
        "steps_per_cycle": 18,
        "description": (
            "Rhythmic generation and collapse of entanglement. "
            "Separable product states swell into deeply correlated "
            "Bell pairs, then disentangle back to independence."
        ),
    },
    "error_correction_rhythm": {
        "state_a": "bell_entangler",
        "state_b": "surface_code_patch",
        "pattern": "sinusoidal",
        "num_cycles": 3,
        "steps_per_cycle": 26,
        "description": (
            "Transition from raw entanglement to structured error-"
            "correcting topology. Models the encoding/decoding cycle "
            "of fault-tolerant quantum computation."
        ),
    },
    "quantum_advantage_toggle": {
        "state_a": "clifford_stabilizer",
        "state_b": "random_supremacy",
        "pattern": "square",
        "num_cycles": 3,
        "steps_per_cycle": 30,
        "description": (
            "Sharp toggle between classically simulable Clifford "
            "circuits and random supremacy-class configurations. "
            "The computational phase boundary as aesthetic rhythm."
        ),
    },
}


# ============================================================================
# PHASE 2.7: VISUAL TYPES (Image-Generation Vocabulary)
# ============================================================================
#
# Each visual type maps to a region of the 5D morphospace and carries
# image-generation-ready keywords with explicit geometric specifications.
# ============================================================================

VISUAL_TYPES: Dict[str, Dict] = {
    "green_spider_lattice": {
        "coords": {
            "spider_density": 0.80,
            "phase_complexity": 0.20,
            "entanglement_depth": 0.60,
            "chromatic_balance": 0.15,
            "flow_determinism": 0.85,
        },
        "keywords": [
            "emerald crystalline lattice of connected nodes",
            "dense geometric grid with green circular vertices",
            "repeating hexagonal spider network at 60-degree intervals",
            "sharp radial edges converging on phase-labeled hubs",
            "monochromatic green palette against matte black ground",
            "precise bilateral symmetry across central vertical axis",
            "thin luminous wires linking each node to four neighbours",
        ],
    },
    "red_spider_web": {
        "coords": {
            "spider_density": 0.65,
            "phase_complexity": 0.40,
            "entanglement_depth": 0.85,
            "chromatic_balance": 0.90,
            "flow_determinism": 0.50,
        },
        "keywords": [
            "crimson filament network with radial measurement arms",
            "red circular nodes emitting fine branching tendrils",
            "X-basis measurement cones opening at 45-degree fans",
            "warm red-to-amber gradient along each wire segment",
            "asymmetric web structure with spiral connectivity",
            "translucent layered depth with nodes at staggered planes",
            "soft diffuse glow around high-degree junction vertices",
        ],
    },
    "phase_gadget_constellation": {
        "coords": {
            "spider_density": 0.55,
            "phase_complexity": 0.90,
            "entanglement_depth": 0.50,
            "chromatic_balance": 0.35,
            "flow_determinism": 0.75,
        },
        "keywords": [
            "iridescent phase-labeled nodes in scattered constellation",
            "angular crystal formations at π/4 rotation increments",
            "prismatic colour shift from green through gold to violet",
            "small tight clusters of three nodes bridged by single wires",
            "phase annotations rendered as luminous arc segments",
            "dark negative space emphasising isolated gadget modules",
            "faceted gem-like vertices refracting internal light",
        ],
    },
    "wire_identity_flow": {
        "coords": {
            "spider_density": 0.08,
            "phase_complexity": 0.05,
            "entanglement_depth": 0.05,
            "chromatic_balance": 0.50,
            "flow_determinism": 0.98,
        },
        "keywords": [
            "minimal parallel horizontal wires with no interruption",
            "ultra-clean white lines on deep black background",
            "uniform 12-pixel spacing between each wire track",
            "perfect left-to-right causal flow with no branching",
            "no nodes or vertices breaking line continuity",
            "hairline precision with anti-aliased edges",
            "austere negative space dominating 90 percent of frame",
        ],
    },
    "entangled_bell_pair": {
        "coords": {
            "spider_density": 0.25,
            "phase_complexity": 0.10,
            "entanglement_depth": 0.95,
            "chromatic_balance": 0.50,
            "flow_determinism": 0.90,
        },
        "keywords": [
            "paired symmetric structures mirrored across horizontal axis",
            "two large nodes connected by thick luminous bridge wire",
            "quantum correlation arc drawn as continuous sine curve",
            "balanced green-red colour split at the midpoint junction",
            "clean bilateral symmetry with minimal surrounding elements",
            "bright focal glow at the entanglement junction vertex",
            "sparse surrounding space emphasising the central bond",
        ],
    },
}


# ============================================================================
# PHASE 2.6 HELPER FUNCTIONS (Layer 2 — deterministic, 0 tokens)
# ============================================================================

def _generate_oscillation(num_steps: int, num_cycles: float, pattern: str) -> np.ndarray:
    """Generate oscillation envelope in [0, 1].  endpoint=False guarantees closure."""
    t = np.linspace(0, 2 * np.pi * num_cycles, num_steps, endpoint=False)
    if pattern == "sinusoidal":
        return 0.5 * (1.0 + np.sin(t))
    elif pattern == "triangular":
        t_norm = (t / (2 * np.pi)) % 1.0
        return np.where(t_norm < 0.5, 2.0 * t_norm, 2.0 * (1.0 - t_norm))
    elif pattern == "square":
        t_norm = (t / (2 * np.pi)) % 1.0
        return np.where(t_norm < 0.5, 0.0, 1.0)
    raise ValueError(f"Unknown oscillation pattern: {pattern}")


def _state_to_vec(state_id: str) -> np.ndarray:
    """Convert canonical state id to 5D vector."""
    s = CANONICAL_STATES[state_id]
    return np.array([s[p] for p in PARAMETER_NAMES])


def _vec_to_dict(vec: np.ndarray) -> Dict[str, float]:
    """Convert 5D vector to parameter dict."""
    return {p: round(float(vec[i]), 6) for i, p in enumerate(PARAMETER_NAMES)}


def _generate_preset_trajectory(preset_cfg: Dict) -> List[Dict[str, float]]:
    """Generate full trajectory for a preset (forced orbit, endpoint=False)."""
    vec_a = _state_to_vec(preset_cfg["state_a"])
    vec_b = _state_to_vec(preset_cfg["state_b"])
    total_steps = preset_cfg["num_cycles"] * preset_cfg["steps_per_cycle"]
    alpha = _generate_oscillation(total_steps, preset_cfg["num_cycles"], preset_cfg["pattern"])
    trajectory = np.outer(1.0 - alpha, vec_a) + np.outer(alpha, vec_b)
    return [_vec_to_dict(trajectory[i]) for i in range(total_steps)]


def _nearest_visual_type(state: Dict[str, float]) -> Tuple[str, float, List[str]]:
    """Find nearest visual type by Euclidean distance in 5D."""
    query = np.array([state.get(p, 0.5) for p in PARAMETER_NAMES])
    best_name, best_dist, best_kw = None, float("inf"), []
    for vt_name, vt in VISUAL_TYPES.items():
        ref = np.array([vt["coords"][p] for p in PARAMETER_NAMES])
        d = float(np.linalg.norm(query - ref))
        if d < best_dist:
            best_name, best_dist, best_kw = vt_name, d, vt["keywords"]
    return best_name, best_dist, best_kw


# ============================================================================
# PHASE 2.6 TOOLS
# ============================================================================

@mcp.tool()
async def get_zx_coordinates(state_id: str) -> dict:
    """
    Get normalized 5D coordinates for a canonical quantum ZX state.

    Returns the position in the quantum ZX morphospace for use in
    cross-domain composition with other Lushy aesthetic domains.

    Args:
        state_id: Canonical state identifier. One of:
            identity_wire, clifford_stabilizer, t_magic_state,
            bell_entangler, surface_code_patch, random_supremacy,
            measurement_pattern

    Returns:
        5D coordinate dict with parameter semantics

    Cost: 0 tokens (dictionary lookup)
    """
    if state_id not in CANONICAL_STATES:
        return {
            "success": False,
            "error": f"Unknown state '{state_id}'",
            "available_states": list(CANONICAL_STATES.keys()),
        }
    coords = CANONICAL_STATES[state_id]
    return {
        "success": True,
        "state_id": state_id,
        "coordinates": coords,
        "parameter_names": PARAMETER_NAMES,
        "parameter_semantics": {
            "spider_density": "0.0 = minimal/identity wires, 1.0 = dense lattice",
            "phase_complexity": "0.0 = Clifford-only, 1.0 = arbitrary phases",
            "entanglement_depth": "0.0 = separable product, 1.0 = volume-law",
            "chromatic_balance": "0.0 = Z-spider (green), 1.0 = X-spider (red)",
            "flow_determinism": "0.0 = non-deterministic, 1.0 = fully causal",
        },
        "cost_tokens": 0,
    }


@mcp.tool()
async def list_zx_rhythmic_presets() -> dict:
    """
    List all Phase 2.6 rhythmic presets for the quantum ZX domain.

    Each preset defines a forced-orbit oscillation between two canonical
    states with a specific period, pattern, and cycle count.

    Returns:
        Preset catalog with period strategy annotations

    Cost: 0 tokens (dictionary access)
    """
    presets_out = {}
    for name, cfg in PHASE26_PRESETS.items():
        presets_out[name] = {
            "state_a": cfg["state_a"],
            "state_b": cfg["state_b"],
            "pattern": cfg["pattern"],
            "num_cycles": cfg["num_cycles"],
            "steps_per_cycle": cfg["steps_per_cycle"],
            "total_steps": cfg["num_cycles"] * cfg["steps_per_cycle"],
            "description": cfg["description"],
        }
    return {
        "success": True,
        "domain": "quantum_zx",
        "total_presets": len(presets_out),
        "presets": presets_out,
        "periods": sorted(set(c["steps_per_cycle"] for c in PHASE26_PRESETS.values())),
        "period_strategy": {
            "14": "gap-filler 12-15, LCM(14,30)=210, novel harmonics",
            "18": "4-domain overlap: nuclear+catastrophe+diatom+quantum",
            "22": "3-domain overlap: catastrophe+heraldic+quantum",
            "26": "gap-filler 25-30, LCM(26,30)=390, novel harmonics",
            "30": "major hub: microscopy+diatom+heraldic+quantum (4-domain lock)",
        },
        "cost_tokens": 0,
    }


@mcp.tool()
async def apply_zx_rhythmic_preset(preset_name: str) -> dict:
    """
    Apply a Phase 2.6 rhythmic preset, generating the complete forced-orbit
    oscillation trajectory.

    Uses endpoint=False on np.linspace to guarantee perfect periodic closure
    with zero numerical drift.

    Args:
        preset_name: One of: optimization_cycle, universality_sweep,
            entanglement_pulse, error_correction_rhythm,
            quantum_advantage_toggle

    Returns:
        Full trajectory as list of 5D coordinate dicts, plus metadata

    Cost: 0 tokens (deterministic NumPy computation)
    """
    if preset_name not in PHASE26_PRESETS:
        return {
            "success": False,
            "error": f"Unknown preset '{preset_name}'",
            "available_presets": list(PHASE26_PRESETS.keys()),
        }
    cfg = PHASE26_PRESETS[preset_name]
    trajectory = _generate_preset_trajectory(cfg)

    # Closure validation: first step of each cycle should repeat
    steps_per_cycle = cfg["steps_per_cycle"]
    closure_drift = 0.0
    if len(trajectory) > steps_per_cycle:
        v0 = np.array([trajectory[0][p] for p in PARAMETER_NAMES])
        vc = np.array([trajectory[steps_per_cycle][p] for p in PARAMETER_NAMES])
        closure_drift = float(np.linalg.norm(v0 - vc))

    return {
        "success": True,
        "preset_name": preset_name,
        "config": {
            "state_a": cfg["state_a"],
            "state_b": cfg["state_b"],
            "pattern": cfg["pattern"],
            "num_cycles": cfg["num_cycles"],
            "steps_per_cycle": steps_per_cycle,
        },
        "trajectory_length": len(trajectory),
        "trajectory": trajectory,
        "closure_validation": {
            "drift": closure_drift,
            "is_closed": closure_drift < 1e-10,
        },
        "cost_tokens": 0,
    }


@mcp.tool()
async def generate_zx_rhythmic_sequence(
    preset_name: str,
    start_step: int = 0,
    num_steps: int = 10,
) -> dict:
    """
    Generate a windowed slice of a rhythmic preset trajectory.

    Useful for stepping through the oscillation in chunks or
    extracting a specific phase region for prompt generation.

    Args:
        preset_name: Preset identifier
        start_step: Starting index in the full trajectory (wraps around)
        num_steps: How many steps to return

    Returns:
        Trajectory slice with per-step visual vocabulary annotations

    Cost: 0 tokens (deterministic computation)
    """
    if preset_name not in PHASE26_PRESETS:
        return {
            "success": False,
            "error": f"Unknown preset '{preset_name}'",
            "available_presets": list(PHASE26_PRESETS.keys()),
        }
    cfg = PHASE26_PRESETS[preset_name]
    full_traj = _generate_preset_trajectory(cfg)
    total = len(full_traj)

    sequence = []
    for i in range(num_steps):
        idx = (start_step + i) % total
        state = full_traj[idx]
        vt_name, vt_dist, vt_kw = _nearest_visual_type(state)
        sequence.append({
            "step": idx,
            "coordinates": state,
            "nearest_visual_type": vt_name,
            "visual_distance": round(vt_dist, 4),
            "keywords": vt_kw,
        })

    return {
        "success": True,
        "preset_name": preset_name,
        "start_step": start_step,
        "num_steps": num_steps,
        "total_trajectory_length": total,
        "sequence": sequence,
        "cost_tokens": 0,
    }


# ============================================================================
# PHASE 2.7 TOOLS: ATTRACTOR VISUALIZATION PROMPT GENERATION
# ============================================================================

@mcp.tool()
async def get_zx_visual_types() -> dict:
    """
    List all visual vocabulary types for the quantum ZX domain.

    Each type maps to a region of the 5D morphospace and carries
    image-generation-ready keywords with explicit geometric specifications.

    Returns:
        Visual type catalog with coordinates and keyword lists

    Cost: 0 tokens (dictionary access)
    """
    types_out = {}
    for vt_name, vt in VISUAL_TYPES.items():
        types_out[vt_name] = {
            "coordinates": vt["coords"],
            "keywords": vt["keywords"],
            "keyword_count": len(vt["keywords"]),
        }
    return {
        "success": True,
        "domain": "quantum_zx",
        "total_visual_types": len(types_out),
        "visual_types": types_out,
        "cost_tokens": 0,
    }


@mcp.tool()
async def extract_zx_visual_vocabulary(
    state: str,
    strength: float = 1.0,
) -> dict:
    """
    Extract visual vocabulary for an arbitrary point in quantum ZX morphospace.

    Finds the nearest visual type by Euclidean distance in 5D and returns
    its image-generation keywords, weighted by strength.

    Args:
        state: JSON string of 5D coordinates, OR a canonical state id
        strength: Weight multiplier [0.0–1.0] for blending in multi-domain
            compositions.  1.0 = full domain contribution.

    Returns:
        Nearest visual type, distance, and weighted keywords

    Cost: 0 tokens (nearest-neighbor lookup)
    """
    import json as _json

    # Accept either a state_id string or a JSON coordinate dict
    if state in CANONICAL_STATES:
        coords = CANONICAL_STATES[state]
    else:
        try:
            coords = _json.loads(state)
        except Exception:
            return {
                "success": False,
                "error": (
                    f"'{state}' is not a canonical state id or valid JSON. "
                    f"Available states: {list(CANONICAL_STATES.keys())}"
                ),
            }

    vt_name, vt_dist, vt_kw = _nearest_visual_type(coords)

    return {
        "success": True,
        "input_coordinates": coords,
        "nearest_visual_type": vt_name,
        "distance": round(vt_dist, 6),
        "strength": strength,
        "keywords": vt_kw,
        "cost_tokens": 0,
    }


@mcp.tool()
async def generate_zx_attractor_visualization_prompt(
    state: str,
    mode: str = "composite",
    additional_context: str = "",
) -> dict:
    """
    Generate an image-generation-ready prompt from quantum ZX morphospace
    coordinates.

    Three prompt modes:
      composite   — single blended prompt combining all keywords
      vocabulary  — returns raw keyword list for external composition
      geometric   — keywords reformulated as explicit spatial directives

    Args:
        state: JSON string of 5D coordinates, OR a canonical state id
        mode: One of composite, vocabulary, geometric
        additional_context: Extra text appended to composite prompts

    Returns:
        Prompt string(s) suitable for Stable Diffusion / DALL-E / Midjourney

    Cost: 0 tokens (deterministic string assembly)
    """
    import json as _json

    if state in CANONICAL_STATES:
        coords = CANONICAL_STATES[state]
        state_label = state
    else:
        try:
            coords = _json.loads(state)
            state_label = "custom"
        except Exception:
            return {
                "success": False,
                "error": (
                    f"'{state}' is not a canonical state id or valid JSON. "
                    f"Available states: {list(CANONICAL_STATES.keys())}"
                ),
            }

    vt_name, vt_dist, vt_kw = _nearest_visual_type(coords)

    if mode == "vocabulary":
        return {
            "success": True,
            "mode": "vocabulary",
            "state_label": state_label,
            "nearest_visual_type": vt_name,
            "distance": round(vt_dist, 6),
            "keywords": vt_kw,
            "cost_tokens": 0,
        }

    elif mode == "geometric":
        # Rewrite keywords as explicit spatial/geometric directives
        geo_kw = []
        for kw in vt_kw:
            geo_kw.append(kw)  # keywords already contain geometric specs
        prompt = (
            "Quantum diagram visualization, technical illustration style. "
            + ". ".join(geo_kw)
        )
        if additional_context:
            prompt += f". {additional_context}"
        return {
            "success": True,
            "mode": "geometric",
            "state_label": state_label,
            "nearest_visual_type": vt_name,
            "prompt": prompt,
            "cost_tokens": 0,
        }

    else:  # composite (default)
        kw_str = ", ".join(vt_kw)
        prompt = (
            f"Quantum ZX-calculus diagram in the style of {vt_name.replace('_', ' ')}. "
            f"{kw_str}."
        )
        if additional_context:
            prompt += f" {additional_context}."
        return {
            "success": True,
            "mode": "composite",
            "state_label": state_label,
            "nearest_visual_type": vt_name,
            "distance": round(vt_dist, 6),
            "prompt": prompt,
            "cost_tokens": 0,
        }


@mcp.tool()
async def compute_zx_state_distance(state_a: str, state_b: str) -> dict:
    """
    Compute Euclidean distance between two points in the quantum ZX
    morphospace.

    Accepts canonical state ids or JSON coordinate dicts.

    Args:
        state_a: First state (id or JSON)
        state_b: Second state (id or JSON)

    Returns:
        Distance and per-parameter deltas

    Cost: 0 tokens (arithmetic)
    """
    import json as _json

    def _resolve(s):
        if s in CANONICAL_STATES:
            return CANONICAL_STATES[s], s
        return _json.loads(s), "custom"

    try:
        ca, la = _resolve(state_a)
        cb, lb = _resolve(state_b)
    except Exception as e:
        return {"success": False, "error": str(e)}

    va = np.array([ca[p] for p in PARAMETER_NAMES])
    vb = np.array([cb[p] for p in PARAMETER_NAMES])
    diff = vb - va
    dist = float(np.linalg.norm(diff))

    return {
        "success": True,
        "state_a": la,
        "state_b": lb,
        "euclidean_distance": round(dist, 6),
        "per_parameter_delta": {p: round(float(diff[i]), 6) for i, p in enumerate(PARAMETER_NAMES)},
        "cost_tokens": 0,
    }


# ============================================================================
# TIER 4D: DOMAIN REGISTRY CONFIG (for cross-domain composition)
# ============================================================================

@mcp.tool()
async def get_zx_domain_registry_config() -> dict:
    """
    Export domain configuration for Tier 4D cross-domain integration.

    Returns everything needed by the domain_registry and composition-graph
    systems to include quantum ZX in multi-domain attractor discovery.

    Returns:
        Complete domain registration data including presets, coordinates,
        parameter names, periods, and visual vocabulary

    Cost: 0 tokens (dictionary assembly)
    """
    presets_export = {}
    for name, cfg in PHASE26_PRESETS.items():
        presets_export[name] = {
            "period": cfg["steps_per_cycle"],
            "state_a_id": cfg["state_a"],
            "state_b_id": cfg["state_b"],
            "pattern": cfg["pattern"],
            "num_cycles": cfg["num_cycles"],
            "description": cfg["description"],
        }

    return {
        "success": True,
        "domain_id": "quantum_zx",
        "display_name": "Quantum ZX-Calculus",
        "mcp_server": "quantum-zx-mcp",
        "parameter_names": PARAMETER_NAMES,
        "canonical_states": CANONICAL_STATES,
        "presets": presets_export,
        "periods": sorted(set(c["steps_per_cycle"] for c in PHASE26_PRESETS.values())),
        "visual_types": {
            name: {"coords": vt["coords"], "keywords": vt["keywords"]}
            for name, vt in VISUAL_TYPES.items()
        },
        "tier4d_period_strategy": {
            "overlap_periods": [18, 22, 30],
            "gap_filling_periods": [14, 26],
            "predicted_emergent_attractors": [
                {
                    "period": 14,
                    "mechanism": "gap-filler 12-15",
                    "expected_basin": "2-5%",
                    "lcm_with_hub_30": 210,
                },
                {
                    "period": 26,
                    "mechanism": "gap-filler 25-30",
                    "expected_basin": "2-5%",
                    "lcm_with_hub_30": 390,
                },
                {
                    "period": 90,
                    "mechanism": "LCM(18,30) harmonic hub",
                    "expected_basin": "3-8%",
                },
            ],
        },
        "cost_tokens": 0,
    }


# ============================================================================
# TOOL: Get Domain Statistics (updated with Phase 2.6/2.7 info)
# ============================================================================

@mcp.tool()
async def get_quantum_domain_info() -> dict:
    """
    Get information about the quantum ZX-calculus domain.
    
    Returns statistics on registered gates, rewrite rules, and capabilities.
    
    Returns:
        Domain statistics and metadata
        
    Cost: 0 tokens (deterministic lookup)
    """
    stats = get_domain_statistics(QUANTUM_DOMAIN)
    
    return {
        "domain_name": QUANTUM_DOMAIN.name,
        "domain_type": QUANTUM_DOMAIN.domain_type.value,
        "version": "2.0.0",
        "statistics": stats,
        "capabilities": {
            "flow_verification": True,
            "automatic_optimization": True,
            "rewrite_rules": len(QUANTUM_DOMAIN.rewrite_rules),
            "cross_domain_transforms": True,
        },
        "phase_2_6_enhancements": {
            "rhythmic_composition": True,
            "parameter_count": len(PARAMETER_NAMES),
            "canonical_states": list(CANONICAL_STATES.keys()),
            "preset_count": len(PHASE26_PRESETS),
            "periods": sorted(set(c["steps_per_cycle"] for c in PHASE26_PRESETS.values())),
        },
        "phase_2_7_enhancements": {
            "attractor_visualization": True,
            "visual_type_count": len(VISUAL_TYPES),
            "visual_types": list(VISUAL_TYPES.keys()),
            "prompt_modes": ["composite", "vocabulary", "geometric"],
        },
        "tier4d_integration": {
            "domain_id": "quantum_zx",
            "overlap_periods": [18, 22, 30],
            "gap_filling_periods": [14, 26],
        },
        "metadata": QUANTUM_DOMAIN.metadata,
        "cost_tokens": 0,
    }


# ============================================================================
# TOOL: Lookup Gate
# ============================================================================

@mcp.tool()
async def lookup_quantum_gate(gate_name: str) -> dict:
    """
    Look up a quantum gate by name.
    
    Args:
        gate_name: Gate identifier (e.g., 'h', 'cx', 't', 'cnot')
        
    Returns:
        Gate definition with ZX-diagram representation
        
    Cost: 0 tokens (dictionary lookup)
    """
    gate_info = lookup_gate(QUANTUM_DOMAIN, gate_name)
    
    if not gate_info:
        return {
            "success": False,
            "error": f"Gate '{gate_name}' not found",
            "available_gates": list(QUANTUM_DOMAIN.objects.keys())
        }
    
    spider_type = get_gate_spider_type(QUANTUM_DOMAIN, gate_name)
    
    return {
        "success": True,
        "gate": gate_info,
        "spider_type": spider_type,
        "zx_representation": f"{spider_type}-spider with phase {gate_info['phase']}π",
        "cost_tokens": 0
    }


# ============================================================================
# TOOL: Analyze Circuit
# ============================================================================

@mcp.tool()
async def analyze_circuit(qasm_code: str) -> dict:
    """
    Analyze quantum circuit using categorical framework.
    
    Provides both generic categorical analysis (flow structure, depth, 
    entanglement) and quantum-specific metrics (Clifford gates, T-count, etc.).
    
    Args:
        qasm_code: OpenQASM 2.0 circuit code
        
    Returns:
        Comprehensive analysis with categorical and quantum metrics
        
    Cost: 0 tokens (deterministic graph analysis)
    """
    # Parse circuit
    circuit = parse_qasm_circuit(qasm_code)
    if not circuit:
        return {
            "success": False,
            "error": "Failed to parse QASM code. Check syntax."
        }
    
    # Use categorical framework analysis
    analysis = analyze_quantum_circuit(circuit, QUANTUM_DOMAIN)
    
    # Extract key metrics
    cat = analysis["categorical_analysis"]
    quant = analysis["quantum_specific"]
    opts = analysis["optimization_recommendations"]
    issues = analysis["potential_issues"]
    
    return {
        "success": True,
        "circuit_info": {
            "num_qubits": circuit.num_qubits,
            "num_gates": len(circuit.gates),
            "gate_sequence": [g[0] for g in circuit.gates[:10]]  # First 10 gates
        },
        "categorical_analysis": {
            "total_nodes": cat["total_nodes"],
            "total_edges": cat["total_edges"],
            "has_flow": cat["has_flow"],
            "has_cycles": cat["has_cycles"],
            "max_depth": cat["max_depth"],
            "entanglement_score": cat["entanglement_score"],
            "deterministic_fraction": cat["deterministic_fraction"]
        },
        "quantum_analysis": {
            "clifford_gates": quant["clifford_gates"],
            "t_gates": quant["t_gates"],
            "two_qubit_gates": quant["two_qubit_gates"],
            "is_clifford_only": quant["is_clifford_only"],
            "estimated_t_count": quant["estimated_t_count"],
            "has_measurements": quant["has_measurements"]
        },
        "recommendations": opts,
        "potential_issues": issues,
        "cost_tokens": 0
    }


# ============================================================================
# TOOL: Verify Flow Structure
# ============================================================================

@mcp.tool()
async def verify_flow_structure(qasm_code: str) -> dict:
    """
    Verify circuit admits valid flow structure (Paper Definition 3.16).
    
    Flow structure proves that:
    - Composition is mathematically valid
    - Errors can be corrected deterministically
    - No circular dependencies exist
    
    This is a formal verification based on the de Felice et al. paper.
    
    Args:
        qasm_code: OpenQASM 2.0 circuit code
        
    Returns:
        Flow verification results with correction structure
        
    Cost: 0 tokens (deterministic graph algorithm)
    """
    circuit = parse_qasm_circuit(qasm_code)
    if not circuit:
        return {
            "success": False,
            "error": "Failed to parse QASM code"
        }
    
    # Convert to composition graph
    graph = circuit_to_composition_graph(circuit, QUANTUM_DOMAIN)
    
    # Compute flow structure (Paper Definition 3.16)
    flow = compute_flow_structure(graph)
    
    if flow:
        is_valid, error_msg = flow.verify(graph)
        
        return {
            "success": True,
            "has_flow": True,
            "is_valid": is_valid,
            "verification": {
                "partial_order": flow.partial_order,
                "correction_sets": {
                    node: list(corrections) 
                    for node, corrections in flow.correction_function.items()
                },
                "preserves": flow.preserves
            },
            "interpretation": {
                "mathematical_validity": "Circuit composition is formally valid" if is_valid else "Circuit has structural issues",
                "error_correction": "Errors can be corrected deterministically" if is_valid else "Error correction not guaranteed",
                "paper_reference": "de Felice et al. (2025), Definition 3.16 (Pauli flow)"
            },
            "error": error_msg,
            "cost_tokens": 0
        }
    else:
        return {
            "success": True,
            "has_flow": False,
            "reason": "Circuit contains cycles or no valid correction pattern exists",
            "interpretation": {
                "issue": "Circuit does not admit flow structure",
                "implication": "May have circular dependencies or invalid composition",
                "suggestion": "Reorder gates to eliminate cycles"
            },
            "cost_tokens": 0
        }


# ============================================================================
# TOOL: Optimize Circuit
# ============================================================================

@mcp.tool()
async def optimize_circuit(qasm_code: str, strategy: str = "full") -> dict:
    """
    Optimize quantum circuit using ZX-calculus rewrite rules.
    
    Applies semantics-preserving transformations to reduce circuit complexity.
    Available strategies:
    - "clifford": Optimize Clifford gates (fusion, cancellation)
    - "t_count": Reduce T-gate count (critical for fault tolerance)
    - "minimize_depth": Reduce circuit depth
    - "maximize_determinism": Increase deterministic operations
    - "full": Apply all applicable optimizations
    
    Args:
        qasm_code: OpenQASM 2.0 circuit code
        strategy: Optimization strategy (default: "full")
        
    Returns:
        Optimized circuit statistics and improvement metrics
        
    Cost: 0 tokens (deterministic pattern matching and graph transformation)
    """
    circuit = parse_qasm_circuit(qasm_code)
    if not circuit:
        return {
            "success": False,
            "error": "Failed to parse QASM code"
        }
    
    # Convert to composition graph
    graph = circuit_to_composition_graph(circuit, QUANTUM_DOMAIN)
    
    # Apply optimization strategy
    engine = RewriteEngine(QUANTUM_DOMAIN)
    
    # Get applicable rules before optimization
    applicable_before = QUANTUM_DOMAIN.get_applicable_rules(graph)
    
    # Optimize based on strategy
    if strategy == "full":
        optimized = engine.apply_all_applicable(graph)
    else:
        optimized = engine.optimize_for_goal(graph, strategy)
    
    # Analyze both graphs
    original_analysis = analyze_composition(graph)
    optimized_analysis = analyze_composition(optimized)
    
    # Calculate improvements
    node_reduction = original_analysis.total_nodes - optimized_analysis.total_nodes
    depth_reduction = original_analysis.max_depth - optimized_analysis.max_depth
    
    return {
        "success": True,
        "strategy": strategy,
        "original": {
            "nodes": original_analysis.total_nodes,
            "edges": original_analysis.total_edges,
            "depth": original_analysis.max_depth,
            "entanglement": original_analysis.entanglement_score
        },
        "optimized": {
            "nodes": optimized_analysis.total_nodes,
            "edges": optimized_analysis.total_edges,
            "depth": optimized_analysis.max_depth,
            "entanglement": optimized_analysis.entanglement_score
        },
        "improvement": {
            "node_reduction": node_reduction,
            "node_reduction_percent": (node_reduction / max(1, original_analysis.total_nodes)) * 100,
            "depth_reduction": depth_reduction,
            "depth_reduction_percent": (depth_reduction / max(1, original_analysis.max_depth)) * 100
        },
        "rewrites_applied": {
            "count": len(applicable_before),
            "rules": [rule.name for rule in applicable_before]
        },
        "cost_tokens": 0
    }


# ============================================================================
# TOOL: Get Available Rewrite Rules
# ============================================================================

@mcp.tool()
async def get_rewrite_rules() -> dict:
    """
    Get all available ZX-calculus rewrite rules.
    
    Returns the catalog of semantics-preserving transformations
    that can optimize quantum circuits.
    
    Returns:
        List of rewrite rules with descriptions and examples
        
    Cost: 0 tokens (list access)
    """
    rules_info = []
    
    for rule in QUANTUM_DOMAIN.rewrite_rules:
        rules_info.append({
            "id": rule.id,
            "name": rule.name,
            "description": rule.description,
            "example": rule.example,
            "preserves_semantics": rule.preserves_semantics,
            "metadata": rule.metadata
        })
    
    return {
        "success": True,
        "total_rules": len(rules_info),
        "rules": rules_info,
        "categories": {
            "spider_fusion": "Merge adjacent spiders of same type",
            "phase_cancellation": "Remove spiders with canceling phases",
            "hadamard_removal": "Eliminate adjacent Hadamard pairs",
            "color_change": "Transform between Z and X basis"
        },
        "reference": "Coecke & Duncan, ZX-calculus completeness (2008+)",
        "cost_tokens": 0
    }


# ============================================================================
# TOOL: List Clifford and T Gates
# ============================================================================

@mcp.tool()
async def get_clifford_t_gates() -> dict:
    """
    Get lists of Clifford gates and T gates.
    
    Clifford gates form the stabilizer group (efficient to simulate).
    T gates provide universal quantum computation (expensive in fault tolerance).
    
    Returns:
        Categorized gate lists with explanations
        
    Cost: 0 tokens (list filter)
    """
    clifford_gates = get_clifford_gates(QUANTUM_DOMAIN)
    t_gates = get_t_gates(QUANTUM_DOMAIN)
    
    return {
        "success": True,
        "clifford_gates": {
            "count": len(clifford_gates),
            "gates": clifford_gates,
            "description": "Stabilizer gates - efficient to simulate classically",
            "examples": ["H", "S", "CNOT", "CZ", "X", "Y", "Z"]
        },
        "t_gates": {
            "count": len(t_gates),
            "gates": t_gates,
            "description": "Non-Clifford gates - required for universal QC, expensive in fault tolerance",
            "examples": ["T", "T†"],
            "note": "T-count is critical metric for fault-tolerant quantum computing"
        },
        "theory": {
            "clifford_theorem": "Clifford gates alone cannot achieve universal quantum computation",
            "universality": "Clifford + T gates form universal gate set",
            "fault_tolerance": "T gates require expensive magic state distillation"
        },
        "cost_tokens": 0
    }


# ============================================================================
# ENTRY POINT FOR FASTMCP CLOUD
# ============================================================================

def handler():
    """
    Entry point for FastMCP Cloud deployment.
    
    FastMCP Cloud expects this function to return the MCP server instance.
    The cloud platform handles the event loop.
    
    Returns:
        FastMCP server instance
    """
    return mcp


# ============================================================================
# LOCAL DEVELOPMENT
# ============================================================================

if __name__ == "__main__":
    # For local testing with server.run()
    mcp.run()
