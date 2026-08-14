"""Single source of truth for the v6 per-type segmentation tokens.

Replaces the 2 generic tokens ([SEG_TUMOR]/[SEG_ORGAN]) with 27 per-type tokens,
so one image can produce multiple DISTINCT masks (each token drives its own
region via the mask decoder). The token id itself carries which structure is
wanted, so GT-mask alignment is identity-based (token id -> its mask), not
positional.

Imported by: model_chat.py, conv_dataset.py, validate.py, mask_facts.py,
factsheet.py, serving/app.py.
"""

# 22 TotalSegmentator organs (5 lung lobes kept separate) -- keys match
# data_processing/agentic/totalseg/extract_slices.TARGET_ORGANS exactly.
TARGET_ORGANS = [
    "liver", "spleen", "pancreas", "kidney_left", "kidney_right", "stomach",
    "gallbladder", "urinary_bladder", "heart", "colon", "small_bowel",
    "duodenum", "esophagus", "trachea", "thyroid_gland", "prostate", "brain",
    "lung_upper_lobe_left", "lung_upper_lobe_right", "lung_lower_lobe_left",
    "lung_lower_lobe_right", "lung_middle_lobe_right",
]

# organ key -> token
TOKEN_FOR_ORGAN = {k: f"[SEG-{k}]" for k in TARGET_ORGANS}

# lesion / lung-field modalities -> token
TOKEN_FOR_MODALITY = {
    "brain_tumors_ct_scan": "[SEG-brain_tumor]",
    "breast_tumors_ct_scan": "[SEG-breast_tumor]",
    "polyp_endoscopy": "[SEG-polyp]",
    "skin_rgbimage": "[SEG-skin_lesion]",
    "lung_CT": "[SEG-lung_fields]",
    "lung_Xray": "[SEG-lung_fields]",
    "total_segmentator": None,  # per-record; resolved from the organ in the path
}

# Ordered list -> stable token ids under tokenizer.add_tokens. Organs first
# (their order fixed by TARGET_ORGANS), then the 4 lesions, then lung_fields.
SEG_TOKENS = (
    [TOKEN_FOR_ORGAN[k] for k in TARGET_ORGANS]
    + ["[SEG-brain_tumor]", "[SEG-breast_tumor]", "[SEG-polyp]", "[SEG-skin_lesion]"]
    + ["[SEG-lung_fields]"]
)
assert len(SEG_TOKENS) == 27, len(SEG_TOKENS)
assert len(set(SEG_TOKENS)) == 27

# Rare organs that don't co-occur on abdominal slices -> oversample in extraction.
RARE_ORGANS = ["thyroid_gland", "prostate", "brain", "esophagus"]

# Back-compat alias so callers importing the old scalar name don't break.
SEG_TOKEN = SEG_TOKENS[0]


def _modality_of(rel_path):
    return str(rel_path).replace("\\", "/").split("/")[0]


def _organ_of(rel_path):
    """total_segmentator masks are named sNNNN_z{z}_{organ}.png or sNNNN_{organ}.png."""
    import os
    name = os.path.splitext(os.path.basename(str(rel_path)))[0]
    # strip case + optional z-slice prefix: s0001[_z123]_organ
    parts = name.split("_")
    # find the organ by matching the longest TARGET_ORGANS suffix
    for k in sorted(TARGET_ORGANS, key=len, reverse=True):
        if name.endswith("_" + k):
            return k
    return parts[-1]


def seg_token_for(rel_mask_path):
    """Per-record token: organ-specific for total_segmentator, else per-modality."""
    modality = _modality_of(rel_mask_path)
    if modality == "total_segmentator":
        return TOKEN_FOR_ORGAN[_organ_of(rel_mask_path)]
    return TOKEN_FOR_MODALITY.get(modality)


def organ_name(token):
    """Human-readable name for a token, for serving legends."""
    key = token.replace("[SEG-", "").rstrip("]")
    return key.replace("_", " ")


# Human-readable request names (organ key -> phrase used in user turns). Keeps
# laterality natural ("left kidney", not "kidney left") and expands lung lobes.
def organ_display(key):
    special = {
        "kidney_left": "left kidney", "kidney_right": "right kidney",
        "lung_upper_lobe_left": "left upper lung lobe",
        "lung_upper_lobe_right": "right upper lung lobe",
        "lung_lower_lobe_left": "left lower lung lobe",
        "lung_lower_lobe_right": "right lower lung lobe",
        "lung_middle_lobe_right": "right middle lung lobe",
        "lung_fields": "lung fields", "brain_tumor": "brain tumor",
        "breast_tumor": "breast tumor", "skin_lesion": "skin lesion",
        "urinary_bladder": "urinary bladder", "thyroid_gland": "thyroid gland",
        "small_bowel": "small bowel",
    }
    return special.get(key, key.replace("_", " "))
