# [kaggle-dstl-satellite-imagery-feature-detection](https://www.kaggle.com/c/dstl-satellite-imagery-feature-detection)
Scripts for the my 6 place solution with [MXNet](https://github.com/dmlc/mxnet).  
Key features:
- Unet-like network architecture with multiple input branches(A, M, P channels)
- Adaptive crop sampler, based on the performance of the network on different classes, updated each epoch.
