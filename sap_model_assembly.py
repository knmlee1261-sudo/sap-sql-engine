"""
SAP Semantic Model Assembly Script
-----------------------------------
Reassembles the per-module JSON files into the full sap_semantic_model.json.
Run this after editing any individual module file to regenerate the combined model.

Usage:
    python sap_model_assembly.py
"""

import json
import os
import glob

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

def assemble_model():
    # 1. Load shared metadata
    with open(os.path.join(BASE_DIR, "sap_model_shared.json")) as f:
        shared = json.load(f)

    # 2. Build the full model structure
    model = {
        "model_metadata": shared["model_metadata"],
        "shared_reference_objects": shared.get("shared_reference_objects", {}),
        "modules": {},
        "cross_module_relationships": shared.get("cross_module_relationships", []),
        "nl_query_patterns": shared.get("nl_query_patterns", []),
        "sql_generation_guidelines": shared.get("sql_generation_guidelines", [])
    }

    # 3. Load each module file
    module_files = sorted(glob.glob(os.path.join(BASE_DIR, "sap_model_fi_*.json")) +
                          glob.glob(os.path.join(BASE_DIR, "sap_model_co.json")) +
                          glob.glob(os.path.join(BASE_DIR, "sap_model_mm.json")) +
                          glob.glob(os.path.join(BASE_DIR, "sap_model_sd.json")) +
                          glob.glob(os.path.join(BASE_DIR, "sap_model_hr.json")) +
                          glob.glob(os.path.join(BASE_DIR, "sap_model_pay.json")) +
                          glob.glob(os.path.join(BASE_DIR, "sap_model_ben.json")) +
                          glob.glob(os.path.join(BASE_DIR, "sap_model_pm.json")))

    modules_loaded = []
    total_tables = 0
    total_columns = 0

    for mf in module_files:
        with open(mf) as f:
            mod = json.load(f)

        mod_key = mod["module_key"]
        # Get module data from the nested structure (under mod_key) or root level
        mod_data = mod.get(mod_key, mod)

        model["modules"][mod_key] = {
            "module_name": mod_data.get("module_name", mod.get("module_name", "")),
            "description": mod_data.get("description", mod.get("description", "")),
            "application_short_name": mod_data.get("application_short_name", mod.get("application_short_name", "")),
            "business_objects": mod_data.get("business_objects", mod.get("business_objects", {}))
        }

        # Count stats
        business_objects = model["modules"][mod_key]["business_objects"]
        tables = sum(len(bo.get("tables", {})) for bo in business_objects.values())
        cols = sum(len(t.get("business_columns", []))
                   for bo in business_objects.values()
                   for t in bo.get("tables", {}).values())
        total_tables += tables
        total_columns += cols
        modules_loaded.append(mod_key)

    # 4. Update metadata
    model["model_metadata"]["modules_covered"] = modules_loaded

    # 5. Write combined model
    output_path = os.path.join(BASE_DIR, "sap_semantic_model.json")
    with open(output_path, "w") as f:
        json.dump(model, f, indent=2)

    print(f"Assembled SAP semantic model:")
    print(f"  Modules: {len(modules_loaded)} — {', '.join(modules_loaded)}")
    print(f"  Tables: {total_tables}")
    print(f"  Columns: {total_columns}")
    print(f"  Cross-module relationships: {len(model['cross_module_relationships'])}")
    print(f"  Query patterns: {len(model['nl_query_patterns'])}")
    print(f"  SQL guidelines: {len(model['sql_generation_guidelines'])}")
    print(f"  Output: {output_path} ({os.path.getsize(output_path):,} bytes)")


if __name__ == "__main__":
    assemble_model()
