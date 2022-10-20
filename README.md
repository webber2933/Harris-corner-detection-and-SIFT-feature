# HW1

**Part 1. Harris Corner Detection**

a. Gaussian Smooth

首先用kernel size和sigma做出一個gaussian kernel
```
# make gaussian kernel
def gaussian_kernels(size,sigma):
    kernel = np.zeros((size,size),dtype = np.float64)
    bound = size//2
    
    for x in range(-bound,size - bound):
        for y in range(-bound,size - bound):
            kernel[y+bound,x+bound] = np.exp( -(x**2 + y**2) / (2 * sigma**2) )
    
    kernel /= (2 * np.pi * sigma**2 )
    kernel /= kernel.sum()
    return kernel
```

接著使用signal.convolve2d將gaussian kernel和image做convolution
```
smooth_img1_size5 = signal.convolve2d(
    img1, 
    gaussian_kernels(5,5), 
    mode='same', boundary='fill', fillvalue=0
)
```

b. Intensity Gradient (Sobel edge detection)

首先先定義出sobel filter
```
filter_gx = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])
filter_gy = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]])
```

接著分別拿這兩個filter和gaussian smooth過後的image做convolution
得到x和y方向的gradient (gx,gy)

並透過公式取得gradient magnitude，再將magnitude的值縮放在0-255之間
```
# gradient magnitude image
gradient_magnitude = np.sqrt(np.square(gx) + np.square(gy))
# visualize the gradient magnitude image
gradient_magnitude = 255.0 * (gradient_magnitude-gradient_magnitude.min())/(gradient_magnitude.max()-gradient_magnitude.min())
gradient_magnitude = np.round(gradient_magnitude)
gradient_magnitude = gradient_magnitude.astype(np.uint8)
```

在gradient direction的部分使用HSV去視覺化算出來的角度
```
gradient_direction = np.arctan2(gy,gx)
h,w = image.shape
hsv = np.zeros((h, w, 3))
hsv[..., 0] = (gradient_direction + np.pi) / (2 * np.pi)
hsv[..., 1] = np.ones((h, w))
hsv[..., 2] = gradient_magnitude
```

c. Structure Tensor

這個function會拿剛剛算出來的gx,gy,window_size(3或5),threshold(R值threshold)當作參數，首先使用gx,gy算出(gxgx),(gygy),(gxgy)

```
xx_grad = gx * gx
yy_grad = gy * gy
xy_grad = gx * gy
```

接著透過一個window滑動去找出每個pixel的structure tensor
```
window_x = xx_grad[i-pad : i+pad+1 , j-pad : j+pad+1]
window_y = yy_grad[i-pad : i+pad+1 , j-pad : j+pad+1]
window_xy = xy_grad[i-pad : i+pad+1 , j-pad : j+pad+1]
sum_xx = np.sum(window_x,dtype = np.int64)
sum_yy = np.sum(window_y,dtype = np.int64)
sum_xy = np.sum(window_xy,dtype = np.int64)
```

算出這個structure tensor的smaller eigenvalue以及R，如果這個pixel的R<threshold，則將這個pixel的smaller eigenvalue視為0

d. Non-maximal Suppression

這邊會在上一步得到的smaller eigenvalue image上做NMS，首先將這些eigenvalue的值從大排到小
```
for i in range(int(eigenvalue.shape[0])):
    for j in range(int(eigenvalue.shape[1])):
        if eigenvalue[i][j] > 0:
            L.append([i,j,eigenvalue[i][j]])

sorted_L = sorted(L, key = lambda x: x[2], reverse = True)
```

使用一個list來記錄NMS之後保留的點，每一次從目前最大eigenvalue的點開始，如果這個點附近已經有其他點被保留，那表示這個點並不是這個區域的local maximum，反之才保留在list裡面

```
for i in sorted_L :
    too_close = False
    for j in final_L :
        if math.sqrt((j[0] - i[0])**2 + (j[1] - i[1])**2) <= dis :
            too_close = True
            break
    if not too_close:
        final_L.append(i[:-1])
        x.append(i[1])
        y.append(i[0])
```

**Part 2. SIFT interest point detection and matching**

首先使用cv2的build-in function在兩張圖各找到100個interest points
```
sift = cv2.xfeatures2d.SIFT_create(100)
keypoints1, descriptors1 = sift.detectAndCompute(img1, None)
keypoints2, descriptors2 = sift.detectAndCompute(img2, None)
```

接著定義計算距離的公式，這邊使用Euclidean distance，算出兩個feature vector之間的dist
```
def Euclidean_dist(des1,des2):
    sum_sq = np.sum(np.square(des1 - des2))
    return np.sqrt(sum_sq)
```

在nearest-neighbor matching的部分，會拿第一張圖的每個feature vector去和第二張圖裡面所有feature vector計算dist，並取最小的dist視為matching

```
for i in range(len(descriptors1)):
    all_dist = []
    ref_descriptor = descriptors1[i]
    for j in range(len(descriptors2)):
        dist = Euclidean_dist(ref_descriptor,descriptors2[j])
        all_dist.append([j,dist])

    all_dist = sorted(all_dist,key = lambda x: x[1])

    matches.append([i,all_dist[0][0]])
```

在第二部分，為了降低mis-match，我在兩張圖各自找了2000個interest points，並將這2000個macth的dist做排序，只取前1/4最小dist的match。
而在這些match中，為了減少ambiguous matching，會去檢查interest point的第二小的dist是不是和最小的dist差很多，如果差夠多的話才視為good match
```
# ratio test
for match in matches:
    if match[2] / match[3] < threshold:
        best_matches.append([match[0],match[1]])
```