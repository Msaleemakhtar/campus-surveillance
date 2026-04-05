import cv2
import numpy as np

# Zone polygons per camera — (x, y) pixel coordinates at 640×480 (or 640×360 for outdoor)
# Each camera has a dict of zone_name → polygon points
_ZONES: dict[str, dict[str, list[tuple[int, int]]]] = {
    "cam01": {"Main Entrance": [(0, 0), (640, 0), (640, 480), (0, 480)]},
    "cam02": {"Hallway A":     [(0, 0), (640, 0), (640, 480), (0, 480)]},
    "cam03": {"Hallway B":     [(0, 0), (640, 0), (640, 480), (0, 480)]},
    "cam04": {"Library":       [(0, 0), (640, 0), (640, 480), (0, 480)]},
    "cam05": {"Cafeteria":     [(0, 0), (640, 0), (640, 480), (0, 480)]},
    "cam06": {"Gym Entrance":  [(0, 0), (640, 0), (640, 480), (0, 480)]},
    "cam07": {"Admin":         [(0, 0), (640, 0), (640, 480), (0, 480)]},
    "cam08": {"Stairwell":     [(0, 0), (640, 0), (640, 480), (0, 480)]},
    "cam09": {"Classroom Exit":[(0, 0), (640, 0), (640, 480), (0, 480)]},
    "cam10": {"Reception":     [(0, 0), (640, 0), (640, 480), (0, 480)]},
    "cam11": {"Main Gate":     [(0, 0), (640, 0), (640, 360), (0, 360)]},
    "cam12": {"Parking A":     [(0, 0), (640, 0), (640, 360), (0, 360)]},
    "cam13": {"Parking B":     [(0, 0), (640, 0), (640, 360), (0, 360)]},
    "cam14": {"Side Entrance": [(0, 0), (640, 0), (640, 360), (0, 360)]},
    "cam15": {"Sports Ground": [(0, 0), (640, 0), (640, 480), (0, 480)]},
    "cam16": {"Pathway A":     [(0, 0), (640, 0), (640, 480), (0, 480)]},
    "cam17": {"Pathway B":     [(0, 0), (640, 0), (640, 480), (0, 480)]},
    "cam18": {"Building Exit": [(0, 0), (640, 0), (640, 480), (0, 480)]},
    "cam19": {"Threat Zone":   [(0, 0), (640, 0), (640, 480), (0, 480)]},
    "cam20": {"Intrusion Zone":[(0, 0), (640, 0), (640, 480), (0, 480)]},
}


def get_zone_for_point(cam_id: str, x: int, y: int) -> str | None:
    """Return zone name if point (x, y) is inside any zone polygon for this camera."""
    zones = _ZONES.get(cam_id, {})
    for zone_name, points in zones.items():
        polygon = np.array(points, dtype=np.int32)
        if cv2.pointPolygonTest(polygon, (float(x), float(y)), measureDist=False) >= 0:
            return zone_name
    return None


def get_zones(cam_id: str) -> dict[str, list[tuple[int, int]]]:
    return _ZONES.get(cam_id, {})
