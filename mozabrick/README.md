# Command Line Usage

MozaBrick includes a command-line interface (CLI) for easy processing of images without writing code. 

## Basic Usage

```bash
python -m mozabrick.cli  ./examples/input/swamp.jpg -o ./examples/output/
```

This will process the image with default settings (32x32 pixel panels in a 2x2 layout) and save all outputs to the `output` directory.

## Command Line Options

```
usage: mozabrick_cli.py [-h] [-o OUTPUT_DIR] [-s PANEL_SIZE] [-r ROWS] [-c COLS]
                        [--target-size TARGET_SIZE]
                        [--panel-target-size PANEL_TARGET_SIZE] [--dither]
                        [--dither-mask DITHER_MASK] [--skip-panels]
                        [--skip-instructions]
                        input_image

MozaBrick - Create buildable pixel art mosaics from images

positional arguments:
  input_image           Path to the input image

optional arguments:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Directory to save output files (default: output)
  -s PANEL_SIZE, --panel-size PANEL_SIZE
                        Size of each panel in pixels (default: 32)
  -r ROWS, --rows ROWS  Number of panel rows (default: 2)
  -c COLS, --cols COLS  Number of panel columns (default: 2)
  --target-size TARGET_SIZE
                        Target size for output image (square) (default: 2048)
  --panel-target-size PANEL_TARGET_SIZE
                        Target size for individual panel images (square) (default: 1024)
  --dither              Apply dithering effect for smoother transitions
  --dither-mask DITHER_MASK
                        Mask image for selective dithering (black allows dithering)
  --skip-panels         Skip saving individual panel images
  --skip-instructions   Skip generating building instructions
```

## Examples

### Processing with Different Panel Layout

Create a 3x4 layout with 16x16 pixel panels
```bash
python -m mozabrick.cli portrait.jpg -r 3 -c 4 -s 16
```

### Apply Dithering
The dithering is quite wonky and poorly tested. May only be applicable for bigger Mozabrick sets
```bash
python -m mozabrick.cli landscape.jpg --dither
```

### Selective Dithering with Mask
Apply dithering only in specific areas defined by a mask. The mask is an image file with black areas marking the areas where dithering will be applied.

```bash
python -m mozabrick.cli portrait.jpg --dither --dither-mask dither_mask.png
```

### Custom Output Directory

```bash
# Save all outputs to a custom directory
python -m mozabrick.cli portrait.jpg -o my_mosaics
```

### Generate Only Main Mosaic Image
Skip generating panels and instructions

```bash
python -m mozabrick.cli quick.jpg --skip-panels --skip-instructions
```

### Custom Image Sizes
Set custom target sizes for output images

```bash
python -m mozabrick.cli portrait.jpg --target-size 4096 --panel-target-size 2048
```
