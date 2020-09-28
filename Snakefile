

import os
from glob import glob

PATHWAY_NAMES_FILE = os.path.join("data","paradigm-scripts-master","pathways","pathways_v1","names.tab")
PATHWAY_FILES = glob(os.path.join("data","paradigm-scripts-master","pathways","pathways_v1","pid_*_pathway.tab"))
print(PATHWAY_NAMES_FILE)
print(PATHWAY_FILES)
TEMP_DIR = "temp"
PATHWAYS_JSON = os.path.join(TEMP_DIR, "processed_pathways.json")
print(PATHWAYS_JSON)


rule all:
    input:
        PATHWAYS_JSON


rule pathways_to_json:
    input:
        pwy_names=PATHWAY_NAMES_FILE,
        pwy_files=PATHWAY_FILES
    output:
        PATHWAYS_JSON 
    shell:
        "python scripts/pathway_to_json.py {input.pwy_names} {input.pwy_files} {output}"


