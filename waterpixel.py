#%% Imports
from cmath import sqrt
import numpy as np
import matplotlib.pyplot as plt
from skimage.io import imread
from skimage import color
from scipy import ndimage

#%% Gradient
def interpolationbilineaire(ima,l,c):
    l,c
    l1=l-np.floor(l)
    l2=np.ceil(l)-l
    c1=c-np.floor(c)
    c2=np.ceil(c)-c   
    ll=np.uint32(np.floor(l))
    cc=np.uint32(np.floor(c))
    val=ima[ll,cc]*l2*c2+ima[ll+1,cc]*l1*c2+ima[ll,cc+1]*l2*c1+ima[ll+1,cc+1]*l1*c1
    return val 
def maximaDirectionGradient(gradx,grady):
    nl,nc=gradx.shape
    norme=np.sqrt(gradx*gradx+grady*grady)+0.1
    gradx=np.divide(gradx,norme)
    grady=np.divide(grady,norme)
    contours=np.zeros((nl,nc),dtype=int);
    for i in range(1,nl-1):
        for j in range(1,nc-1):
            G1=interpolationbilineaire(norme,i+grady[i,j],j+gradx[i,j]); 
            G2=interpolationbilineaire(norme,i-grady[i,j],j-gradx[i,j]); 
            if norme[i,j]>=G1 and norme[i,j]>=G2:
                contours[i,j]=1
            else:
                contours[i,j]=0
    return contours

def computeGradient(ima):
    gradx_sobel = ndimage.sobel(ima, axis=0, mode="constant")
    grady_sobel = ndimage.sobel(ima, axis=1, mode="constant")
    gradnorm_sobel = np.sqrt(gradx_sobel*gradx_sobel+grady_sobel*grady_sobel)
    graddir_sobel = np.arctan2(grady_sobel,gradx_sobel)
    return gradnorm_sobel, gradx_sobel, grady_sobel, graddir_sobel

def computeContours(ima, seuilnorme = -1):
    gradnorm_sobel, gradx_sobel, grady_sobel, _ = computeGradient(ima)
    gradnorm_sobel_max = np.uint8(maximaDirectionGradient(gradx_sobel, grady_sobel))
    contours=(gradnorm_sobel>seuilnorme)*gradnorm_sobel_max
    return contours

def demoContours(ima, seuilnorme, prefix):
    ima = ndimage.gaussian_filter(ima, sigma=0)
    plt.figure('Image '+prefix)
    plt.axis('off')
    plt.imshow(ima, cmap='gray')
    contours = computeContours(ima, seuilnorme)
    plt.figure("Contours "+prefix)
    plt.axis('off')
    plt.imshow(contours, cmap='gray')

#%% Demo Gradient

ima = imread('rocks.tif')
ima = np.array(ima, dtype=np.float64) / 255
ima = color.rgb2gray(ima)
demoContours(ima, 0.7, "rocks")

ima = imread('cell.tif')
ima = np.array(ima, dtype=np.float64) / 255
demoContours(ima, 0.1,"cell")

#%% Chanfrein distance map

def minDistMask(mask, dist, x, y):
    # x -> col index
    # y -> line index
    # mask center must not be infinite
    mhalfj = int(mask.shape[0]/2)
    mhalfi = int(mask.shape[1]/2)
    min = mask[mhalfj, mhalfi] + dist[y][x] # init at mask center
    for j in range(mask.shape[0]): # line index
        for i in range(mask.shape[1]): # col index
            if(y+j < dist.shape[0] and x+i < dist.shape[1]):
                # ignore mask value of inf
                if mask[j][i] != np.inf :
                    di = i - mhalfj
                    dj = j - mhalfi
                    curr = mask[j, i] + dist[y+dj][x+di]
                    if curr < min:
                        min = curr
    return min

def applyMaskFront(d, mask):
    for y in range(d.shape[0]): # line index
        for x in range(d.shape[1]): # col index
            d[y][x] = minDistMask(mask, d, x, y)
    return d
def applyMaskBack(d, mask):
    for y in range(d.shape[0]-1, -1, -1): # line index
        for x in range(d.shape[1]-1, -1, -1): # col index
            d[y][x] = minDistMask(mask, d, x, y)
    return d

def setCentersHexa(m, centercountx, centercounty):
    my = m.shape[0]
    mx = m.shape[1]
    centerdisty = int(my/centercounty)-1
    centerdistx = int(mx/centercountx)-1
    for j in range(0, centercounty+1):
        decalage = 0
        if j%2 == 0 :
            decalage = 1
        for i in range(0, centercountx+1):
            if i%2==0 and (i+decalage)*centerdistx<mx:
                m[j*centerdisty][(i+decalage)*centerdistx] = 0

    return centerdistx, centerdisty

# mask center must not be infinite
# mask infinite coefs are ignored
mask4Front = np.array([
    [1,         1,          1],
    [1,         0,          np.inf],
    [np.inf,    np.inf,     np.inf]])
mask4Back = np.array([
    [np.inf,    np.inf,     np.inf],
    [np.inf,    0,          1],
    [1,         1,          1]])
maskSFront = np.array([
    [np.inf,    11,         np.inf,     11,         np.inf],
    [11,        7,          5,          7,          11],
    [np.inf,    5,          0,          np.inf,     np.inf],
    [np.inf,    np.inf,     np.inf,     np.inf,     np.inf],
    [np.inf,    np.inf,     np.inf,     np.inf,     np.inf]])
maskSBack = np.array([
    [np.inf,    np.inf,     np.inf,     np.inf,     np.inf],
    [np.inf,    np.inf,     np.inf,     np.inf,     np.inf],
    [np.inf,    np.inf,     0,          5,     np.inf],
    [11,        7,          5,          7,          11],
    [np.inf,    11,         np.inf,     11,         np.inf]])

def distanceMap(dist):
    dist = applyMaskFront(dist, maskSFront)
    dist = applyMaskBack(dist, maskSBack)
    return dist

def distanceMapIma(ima, centercountx, centercounty):
    dist = np.zeros((ima.shape[0], ima.shape[1]))
    dist.fill(np.inf)
    centerdistx, centerdisty = setCentersHexa(dist, centercountx, centercounty)
    dist = distanceMap(dist)
    return dist, centerdistx, centerdisty

# %% Demo distance map

ima = imread('rocks.tif')
ima = np.array(ima, dtype=np.float64) / 255
ima = color.rgb2gray(ima)

dist, centerdistx, centerdisty = distanceMapIma(ima, 16, 16)
seuil = (centerdistx + centerdisty)*5/3
dist = (dist < seuil)

plt.figure()
plt.title("Distance to center")
plt.imshow(dist, cmap='gray')

# %% Tagging strongly connected components

def searchTagsInPast(map, i, j, type='8'):
    # type = '8' or '4' (4 connected or 8 connected)
    tags = []
    if i-1>=0 and map[j][i-1] > 0:
        tags.append(map[j][i-1])
    if j-1>=0 and map[j-1][i] > 0:
        tags.append(map[j-1][i])
    if type=='8':
        if i-1>=0 and j-1>=0 and map[j-1][i-1] > 0:
            tags.append(map[j-1][i-1])
        if i+1<map.shape[1] and j-1>=0 and map[j-1][i+1] > 0:
            tags.append(map[j-1][i+1])
    return tags

def tagging(im, type='8'):
    # tags are integers starting at 1
    currtag = 0
    tagmap = np.zeros((im.shape[0], im.shape[1]))
    tagmap.fill(0)
    # Descent
    for j in range(im.shape[0]):
        for i in range(im.shape[1]):
            if im[j][i] > 0 :
                # Look in the pas (pixels up and left)
                pasttags = searchTagsInPast(tagmap, i, j, type)
                if len(pasttags) == 0:
                    currtag += 1
                    tagmap[j][i] = currtag
                elif len(pasttags) == 1:
                    tagmap[j][i] = pasttags[0]
                elif len(pasttags) > 1:
                    tagmap[j][i] = min(pasttags)
    # Ascent ?
    return tagmap

#%% Demo taggin
tagmap = tagging(dist)
print(tagmap)
plt.figure()
plt.title("Distance to center")
plt.imshow(tagmap)

# %% Water pixels

ima = imread('rocks.tif')
ima = np.array(ima, dtype=np.float64) / 255
ima = color.rgb2gray(ima)

# Gradient
gradient, _, _, _ = computeGradient(ima)

# Cells
distmap, centerdistx, centerdisty = distanceMapIma(ima, 16, 16)
distmapseuil = (centerdistx + centerdisty)*5/3
cells = (distmap < distmapseuil)

# Cells tags
tagmap = tagging(cells)

#
cells = (distmap > distmapseuil)
cells = cells*np.amax(gradient)
W = (gradient < cells)*cells + (gradient > cells)*gradient

plt.figure()
plt.title("Gradient")
plt.imshow(gradient, cmap='gray')

plt.figure()
plt.title("Cells")
plt.imshow(cells, cmap='gray')

plt.figure()
plt.title("Tags")
plt.imshow(tagmap)

plt.figure()
plt.title("W")
plt.imshow(W, cmap='gray')

# %%
