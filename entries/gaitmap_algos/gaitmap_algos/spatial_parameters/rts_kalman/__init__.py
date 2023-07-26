shared_metadata = {
    "short_description": "Kalman filter with RTS smoothing",
    "long_description": "Spatial parameters are estimated by a Error Tracking Kalman Filter (ETKF) with "
    "Rauch-Tung-Striebel (RTS) smoothing. ",
    "references": [
        "https://ieeexplore.ieee.org/document/6418869",
        "http://www.mdpi.com/1424-8220/17/4/825",
        "https://arxiv.org/abs/1711.02508",
    ],
    "code_authors": ["MaD-DiGait"],
    "algorithm_authors": ["See references"],
    "implementation_url": "https://github.com/mad-lab-fau/gaitmap",
}

default_metadata = {
    **shared_metadata,
    "long_description": f"{shared_metadata['long_description']} "
    "For ZUPT detection this method uses a simple threshold on the norm of the gyro data.",
}

improved_zupt_default_metadata = {
    **shared_metadata,
    "long_description": f"{shared_metadata['long_description']} "
    "For ZUPT detection this method uses a more advanced ZUPT detector that forces the existence of one ZUPT "
    "event per stride. "
    "This can help with fast walks, where a simple threshold based ZUPT detector tuned for normal speeds might not "
    "be able to detect ZUPT events.",
}
