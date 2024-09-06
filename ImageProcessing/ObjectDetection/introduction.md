

1. [Co-DETR](https://github.com/Sense-X/Co-DETR)

Co-DETR refers to DETRs with Collaborative Hybrid Assignments Training. So, in order to understand Co-DETR, go to understand DETR first.

1.1 DETR
![Alt text](images/2_rcnn_detr.png)
Transformer based object detection. A successful Vision Transformer.
Key points
CNN -> [batch, 2048, 16, 16] -> [batch, 256, 16, 16], positional embedding [batch, 256, 16, 16]. 
Output: two tuple, one is bbox (center_x, center_y, h, w), another is class index. Background as len(all_categories)+1
Lost function: permutation based on Hungarian algorithm.

Issues: (1) small object detection requires higher resolution. (2) DETR requires many more training epochs to converge compared to modern object detectors (because of uniformly spread attention initially during training). (3) Q is not stable cross different epoch.

Youtube videos
i. [DETR: End-to-End Object Detection with Transformers | Paper Explained](https://www.youtube.com/watch?v=BNx-wno-0-g)

1.2 Deformable DETR

Youtube videos
[Deformable DETR | Lecture 38 (Part 3) | Applied Deep Learning (Supplementary)](https://www.youtube.com/watch?v=al1JXZTBIfU)
1.3 Co-DETR

![Alt text](images/fig1_codetr_framework.png)
