#!/usr/bin/env python3
"""
MozaBrick Command Line Interface

A command line tool for creating pixel art mosaics that can be built with physical bricks.
"""

import argparse
import os
from mozabrick import MozabrickProcessor, MozabrickInstructionExporter

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="MozaBrick - Create buildable pixel art mosaics from images",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        "input_image",
        help="Path to the input image"
    )
    
    parser.add_argument(
        "-o", "--output-dir",
        default="output",
        help="Directory to save output files"
    )
    
    parser.add_argument(
        "-s", "--panel-size",
        type=int,
        default=32,
        help="Size of each panel in pixels"
    )
    
    parser.add_argument(
        "-r", "--rows",
        type=int,
        default=2,
        help="Number of panel rows"
    )
    
    parser.add_argument(
        "-c", "--cols",
        type=int,
        default=2,
        help="Number of panel columns"
    )
    
    parser.add_argument(
        "--target-size",
        type=int,
        default=2048,
        help="Target size for output image (square)"
    )
    
    parser.add_argument(
        "--panel-target-size",
        type=int,
        default=1024,
        help="Target size for individual panel images (square)"
    )
    
    parser.add_argument(
        "--dither",
        action="store_true",
        help="Apply dithering effect for smoother transitions"
    )
    
    parser.add_argument(
        "--dither-mask",
        help="Mask image for selective dithering (black allows dithering)"
    )
    
    parser.add_argument(
        "--skip-panels",
        action="store_true",
        help="Skip saving individual panel images"
    )
    
    parser.add_argument(
        "--skip-instructions",
        action="store_true",
        help="Skip generating building instructions"
    )
    
    return parser.parse_args()

def main():
    """Main entry point for the CLI tool."""
    args = parse_args()
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Get base name for output files
    base_name = os.path.splitext(os.path.basename(args.input_image))[0]
    
    # Initialize processor
    processor = MozabrickProcessor(
        panel_size=args.panel_size,
        layout=(args.rows, args.cols)
    )
    
    print(f"Processing image: {args.input_image}")
    print(f"Panel configuration: {args.rows}x{args.cols} panels, {args.panel_size}x{args.panel_size} pixels each")
    
    # Process image to matrix
    matrix = processor.image_to_matrix(args.input_image)
    
    # Apply dithering if requested
    if args.dither:
        print("Applying dithering effect...")
        if args.dither_mask:
            print(f"Using dithering mask: {args.dither_mask}")
            matrix = processor.apply_dithering(matrix, mask_path=args.dither_mask)
        else:
            matrix = processor.apply_dithering(matrix)
    
    # Save full mosaic image
    mosaic_path = os.path.join(args.output_dir, f"{base_name}_mosaic.png")
    print(f"Saving mosaic image: {mosaic_path}")
    processor.matrix_to_image(
        matrix,
        output_path=mosaic_path,
        target_size=(args.target_size, args.target_size)
    )
    
    # Save individual panel images
    if not args.skip_panels:
        panel_base_path = os.path.join(args.output_dir, f"{base_name}_panel")
        print(f"Saving individual panel images: {panel_base_path}_N.png")
        processor.save_panels(
            matrix,
            panel_base_path,
            target_panel_size=args.panel_target_size
        )
    
    # Generate instructions
    if not args.skip_instructions:
        exporter = MozabrickInstructionExporter(processor)
        
        # Text instructions
        text_path = os.path.join(args.output_dir, f"{base_name}_instructions.txt")
        print(f"Generating text instructions: {text_path}")
        exporter.export_text(matrix, text_path)
        
        # Visual instructions
        image_path = os.path.join(args.output_dir, f"{base_name}_instructions.png")
        print(f"Generating visual instructions: {image_path}")
        exporter.export_image(matrix, image_path)
    
    print("Processing complete!")

if __name__ == "__main__":
    main()
