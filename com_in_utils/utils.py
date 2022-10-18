import base64


def decodeImage(imgstring, filename):
    imgdata = base64.b64decode(imgstring)
    with open('research/'+filename, 'wb') as f:
        f.write(imgdata)
        f.close()


def encodeImage(croppedImagePath):
    with open(croppedImagePath, 'rb') as f:
        return base64.b64encode(f.read())
