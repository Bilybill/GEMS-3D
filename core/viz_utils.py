import numpy as np


SUBTYPE_COLORS = {
    1: "red",
    2: "yellow",
    3: "cyan",
    4: "orange",
    5: "gray",
    21: "magenta",
    22: "purple",
    30: "brown",
    31: "green",
    40: "blue",
    41: "teal",
    42: "gold",
    43: "white",
}


def extract_subtype_labels(anomaly, X, Y, Z, fallback_mask, vp_bg=None):
    try:
        if hasattr(anomaly, "subtype_labels"):
            sub_labels = anomaly.subtype_labels(X, Y, Z)
            if getattr(anomaly, "type", "") == "brine_fault_zone":
                viz_sub = np.zeros_like(sub_labels)
                viz_sub[(sub_labels >= 1) & (sub_labels < 10)] = 21
                viz_sub[sub_labels >= 10] = 22
                return viz_sub
            return sub_labels

        if hasattr(anomaly, "build_property_models"):
            multiprops = anomaly.build_property_models(X, Y, Z, vp_bg=vp_bg)
            if getattr(anomaly, "type", "") == "sediment_basement_interface":
                facies_sbi = multiprops.get("facies_label")
                if facies_sbi is not None:
                    viz_sub = np.zeros_like(facies_sbi)
                    viz_sub[facies_sbi == 2] = 30
                    viz_sub[facies_sbi == 3] = 31
                    return viz_sub
            if getattr(anomaly, "type", "") in {"serpentinized_zone", "serpentinized"}:
                serp_sub = multiprops.get("subtype")
                if serp_sub is not None:
                    return serp_sub
    except Exception as exc:
        print(f"Warning: Could not extract specific subtypes: {exc}")

    return fallback_mask
