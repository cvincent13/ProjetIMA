# ProjetIMA
Implementation of the Waterpixels algorithms for super-pixels partition of images.

Base paper:   
Vaïa Machairas, Matthieu Faessel, David Cárdenas-Peña, Théodore Chabardes, Thomas Walter,
et al.. Waterpixels. IEEE Transactions on Image Processing, 2015, 24 (11), pp.3707 - 3716. 10.1109/TIP.2015.2451011. hal-01212760   
https://hal.science/hal-01212760/document

# Project description
This is an implementation of the above mentionned paper. The waterpixel algorithm aims to partition the image into homogenous areas with enforced regularity and adherence
to object boundaries, which are called super-pixels or waterpixels in this case. This algorithm is based on the Watershed transformation applied to a spacially regularized gradient of the image.
This algorithm enables us to control the number of super-pixels and their regularity, while keeping a linear complexity without post-treatment.
