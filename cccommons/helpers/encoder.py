"""
    JSON Encoder for complex types.
"""
from json import JSONEncoder


class DictEncoder(JSONEncoder):
    """
    Implement own JSON Encoder for transfer data to frontend.
    JSONEncoder only allows primitive types instead of objects.
    """

    def default(self, obj: object):
        """

        Args:
            obj:

        Returns:

        """
        return obj.__dict__


def img_to_base64(img) -> str:
    from cv2 import cv2

    return cv2.imencode(".jpg", img)[1].tostring()


def validateJSON(data) -> bool:
    try:
        import json

        json.loads(data)
    except ValueError:
        return False
    return True


__all__ = ["DictEncoder", "img_to_base64", "validateJSON"]
