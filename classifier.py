from typing import Any, Dict, Optional


def classify_severity(features: Optional[Dict[str, Any]]) -> str:
    """
    Classify pothole severity using rule-based thresholds.

    Rules:
    - if max_depth < 0.3 and area < 500: "Shallow"
    - elif max_depth < 0.6: "Moderate"
    - else: "Deep"
    """
    if features is None:
        return "No pothole"

    max_depth = float(features.get("max_depth", 0.0))
    area = float(features.get("area", 0.0))

    if area <= 0:
        return "No pothole"

    if max_depth < 0.3 and area < 500:
        return "Shallow"
    if max_depth < 0.6:
        return "Moderate"
    return "Deep"
