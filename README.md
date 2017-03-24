# Patch-based Texture Synthesis and Transfer

*Revised and improved version by **Salas** ([leVirve](https://github.com/leVirve)@github)*

An implementation of `"Image Quilting and Texture Synthesis, Efros and Freeman", SIGGRAPH 2002`
(published in 2001)

The output depends on two factors : `PatchSize` and `OverlapWidth`
The running time depends on Sample Image dimensions, Desired Image dimensions, ThresholdConstant and PatchSize

## Texture Sythesis

```bash
python PatchBasedSynthesis.py {source_image} {Patch_Size} {Overlap_Width} {Initial_Threshold_error}
```

for example

`python PatchBasedSynthesis.py textures/corn.jpg 30 5 78.0`

## Texture Transfer

```bash
python PatchBasedTextureTransfer.py {texture_image} {source_image} {patch_size} {overlap_width} {init_threshold_scale}
```
for example

`python PatchBasedTextureTransfer.py textures/rice.png src.jpg`

Sample result
- Transfer texture (`textures/rice.png`) onto source image (`src.jpg`)

    - Patch-based

        ![](results/output.png)
    - Patch-based with overlap cost

        ![](results/output_with_overlap.png)
