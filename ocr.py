from io import BytesIO

from PIL import Image
from cnocr import CnOcr


def accurate(b):
    ocr = CnOcr()
    out = ocr.ocr(Image.open(BytesIO(b)))
    result = {}
    words_result = []
    for i in out:
        pos = []
        for j in list(i["position"]):
            pos.append(list(j))
        words = {"words": i["text"], "location": {"left": 0, "top": 0, "width": 0, "height": 0}}
        left = min(pos[0][0], pos[3][0])
        top = min(pos[0][1], pos[1][1])
        width = max(pos[1][0], pos[2][0]) - left
        height = max(pos[2][1], pos[3][1]) - top
        words["location"]["left"] = int(left)
        words["location"]["top"] = int(top)
        words["location"]["width"] = int(width)
        words["location"]["height"] = int(height)
        words_result.append(words)
    result["words_result"] = words_result
    return result
