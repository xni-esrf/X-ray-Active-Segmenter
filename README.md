## X-ray Active Segmenter

This project provides a lightweight Python UI for exploring large 3D volumetric datasets and generate segmentation maps using deep learning.

### Setup
1. Build the container:

```bash
apptainer build container.sif container.def
```

2. Start the container with GPU support:

```bash
apptainer shell --nv container.sif
```

3. Download the foundation model:

`https://drive.google.com/file/d/1A6RbuGG6SqERgDIOmEadFkyy4l_t8Y-C/view?usp=sharing`

4. Store the downloaded model in a directory named `foundation_model`.
5. Keep the model filename exactly `MAE_XNT.cp` (do not rename it).

### Typical Usage
1. Open a raw volume.
2. Create training and validation bounding boxes for the data to segment.
3. Use the 3D brush segmentation tool to fully segment data inside these boxes.
4. Build a dataset from the bounding boxes.
5. Load the model.
6. Train the model.
   - Training time is about 1 hour per training bounding box.
   - At least two GPUs with around 80 GB RAM each are required.
7. Define inference bounding boxes.
8. Apply the trained model to those boxes.
9. Proofread the predictions.
10. Add newly proofread segmented data to the training and validation sets.
11. Train again and repeat the process iteratively.

### Tips
- You can adjust the 3D brush radius. Use a larger radius when high precision is not required.
- Rather than creating a large manual dataset first, it is often faster to annotate a small dataset, train, generate predictions, proofread those predictions, and train again with new generated bounding boxes.
- Features you want to segment should be represented in both the training and validation sets.
- Features you do not want to segment should also be represented in both the training and validation sets.
- For a fixed amount of annotated data, the training set is often larger than the validation set (typically 2x to 10x).
- For each feature, start with easier examples, then progressively add harder cases.
- You can close the application without stopping a running training job. When training finishes, the best checkpoint (best validation score) is saved and can be used for inference.
- Training can be stopped at any point. If at least the first epoch has finished (about 1 hour), the best state reached so far is restored and can be used for inference.
