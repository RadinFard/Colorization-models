## Easily colorize black and white(BGR) images into LAB(colorized) images
- Inspired by : [Richard Zhang/colorization](https://github.com/richzhang/colorization.git)
- Caffemodel : [download immediatly](https://www.youtube.com/redirect?event=video_description&redir_token=QUFFLUhqa0lvZ3Z5VWhOZVJwTGtxbkdRd1lzMUJHS3dvd3xBQ3Jtc0tuVl9ZbGtmRlRkaXRINkNtSDVpSTZzZEN0bWdZaGlsc3Fyb2JnLWxzN1NSd1ROd2R1cXdQVG9mM1lKT0VaM2JORFV5cWdJZklpaHJlSmdxOGdiN185N0RNVTJzSllHZXBZVnhVdzFNejRxdWtiUzNncw&q=https%3A%2F%2Fwww.dropbox.com%2Fs%2Fdx0qvhhp5hbcx7z%2Fcolorization_release_v2.caffemodel%3Fdl%3D1&v=gAmskBNz_Vc)
## How to...
- Let's have a simple review on how to manually colorize.
- Neccesary libraries: numpy & Opencv-python(used as cv2)
1. So we use an import * to import all functions
```python
from numpy import *
from cv2 import *
```
2. Then we store the path of every model and the BW image into a str variable.
```python
prototxt_path = 'models/colorization_deploy_v2.prototxt'
model_path = 'models/colorization_release_v2.caffemodel'    
kernel_path = 'models/pts_in_hull.npy'
image_file_path = 'samples/sample.jpg'
```
3. Then we load them into the readNetFromCaffe(prototxt_path:str, caffeModel:str).
- NOTE: Don't forget the kernel!
```python
net = dnn.readNetFromCaffe(prototxt_path, model_path)
points = load(kernel_path)
```
4. Then we reshape the points variable.
```python
points = points.transpose().reshape(2, 313, 1, 1)
net.getLayer(net.getLayerId('class8_ab')).blobs = [points.astype(float32)]
net.getLayer(net.getLayerId('conv8_313_rh')).blobs = [full([1, 313], 2.686, dtype='float32')]
```
5. We imread and then normalize the image file and convert it to LAB file.
```python
bw_image_file_path = imread(image_file_path)
normalized = bw_image_file_path.astype(float32) / 255.0
lab = cvtColor(normalized, COLOR_BGR2LAB)
```
6. We resize the lab image and split it into L(lightness).
```python
resized = resize(lab, (224, 224))
L = split(resized)[0]
L -= 50
```
7. We use our models to blob from the L.
```python
net.setInput(dnn.blobFromImage(L))
```
8. Now it's time for the AB.
```python
ab = net.forward()[0, :, :, :].transpose((1, 2, 0))
ab = resize(ab, (bw_image_file_path.shape[1], bw_image_file_path.shape[0]))
```
9. Again with our star guest, lightness and at last start the colorizing process
```python
colorized = concatenate((L[:, :, newaxis], ab), axis=2)
colorized = cvtColor(colorized, COLOR_LAB2BGR)
colorized = (255.0 * colorized).astype('uint8')
```
10. Finally we show the results using imshow(winname:str, mat:MatLike)
```python
imshow('BW image_file_path', bw_image_file_path)
imshow('Colorized', colorized)
waitKey(0)
destroyAllWindows()
```
- See the results!
[BW image](https://github.com/RadinFard/Colorization-models/blob/62c624fd13a293d531615dea8ecc0c790f6bb30f/sample_of_BW.png) & [LAB image(https://github.com/RadinFard/Colorization-models/blob/62c624fd13a293d531615dea8ecc0c790f6bb30f/sample_of_colorized.png)
