"""
MozaBrick - A tool for creating pixel art mosaics that can be built with physical bricks.

This module provides classes for processing images into pixelated mosaics with
5 different shading levels and generating assembly instructions.
"""

from PIL import Image
import numpy as np
import math
import os
import cairo

class MozabrickProcessor:
    def __init__(self, panel_size=32, layout=(2, 2)):
        """
        Initialize the processor with configurable panel size and layout.

        Args:
            panel_size (int): Number of pixels in each panel dimension (default: 32)
            layout (tuple): Number of panels in (rows, columns) format (default: 2x2)
        """
        self.panel_size = panel_size
        self.layout = layout
        self.full_matrix_size = (panel_size * layout[0], panel_size * layout[1])

        self.color_map = {
            1: 255,  # White
            2: 192,  # Light gray
            3: 128,  # Medium gray
            4: 64,   # Dark gray
            5: 0     # Black
        }

        # Adjusted thresholds for better gray level separation
        self.reverse_color_map = {
            (200, 255): 1,  # White
            (150, 199): 2,  # Light gray
            (90, 149): 3,   # Medium gray
            (40, 89): 4,    # Dark gray
            (0, 39): 5      # Black
        }

    def calculate_pixel_boundaries(self, image_size):
        """Calculate precise pixel boundaries for the image."""
        pixel_width = image_size[0] / (self.panel_size * self.layout[1])
        pixel_height = image_size[1] / (self.panel_size * self.layout[0])

        # Create arrays of x and y coordinates for pixel boundaries
        x_bounds = [math.floor(i * pixel_width) for i in range(self.panel_size * self.layout[1] + 1)]
        y_bounds = [math.floor(i * pixel_height) for i in range(self.panel_size * self.layout[0] + 1)]

        return x_bounds, y_bounds

    def image_to_matrix(self, image_path):
        """
        Convert an image to a numpy matrix of values 1-5.
        
        Args:
            image_path (str): Path to the input image file
            
        Returns:
            numpy.ndarray: Matrix of values 1-5 representing grayscale levels
        """
        with Image.open(image_path) as img:
            # Preserve original size and convert to grayscale
            original_size = img.size
            img = img.convert('L')

            # Calculate precise pixel boundaries
            x_bounds, y_bounds = self.calculate_pixel_boundaries(original_size)

            # Initialize output matrix
            matrix = np.zeros(self.full_matrix_size, dtype=int)

            # Process each cell in the matrix
            for i in range(len(y_bounds) - 1):
                for j in range(len(x_bounds) - 1):
                    # Get the exact pixel region
                    region = np.array(img.crop((
                        x_bounds[j],  # left
                        y_bounds[i],  # top
                        x_bounds[j + 1],  # right
                        y_bounds[i + 1]  # bottom
                    )))

                    # Calculate median value (more robust than mean)
                    mid_value = np.median(region)

                    # Convert to 1-5 scale using reverse color map
                    for (lower, upper), value in self.reverse_color_map.items():
                        if lower <= mid_value <= upper:
                            matrix[i, j] = value
                            break

            return matrix

    def matrix_to_image(self, matrix, output_path=None, target_size=None):
        """
        Convert a numpy matrix of values 1-5 to a grayscale image.
        
        Args:
            matrix (numpy.ndarray): Matrix of values 1-5
            output_path (str, optional): Path to save the output image
            target_size (tuple, optional): Target size for the output image (width, height)
            
        Returns:
            PIL.Image: The generated image
        """
        if matrix.shape != self.full_matrix_size:
            raise ValueError(f"Matrix must be {self.full_matrix_size[0]}x{self.full_matrix_size[1]}")

        # Create image array at matrix size
        img_array = np.zeros(matrix.shape, dtype=np.uint8)

        # Fill with exact color values
        for i in range(matrix.shape[0]):
            for j in range(matrix.shape[1]):
                value = matrix[i, j]
                if value in self.color_map:
                    img_array[i, j] = self.color_map[value]

        # Create image from array
        img = Image.fromarray(img_array)

        # Scale to target size if provided
        if target_size:
            if target_size[0] <= matrix.shape[1] * 2 and target_size[1] <= matrix.shape[0] * 2:
                # For small sizes, use NEAREST to maintain crisp pixels
                img = img.resize(target_size, Image.Resampling.NEAREST)
            else:
                # For large sizes (like 2043x2043), use high-quality scaling
                scale = 8
                img_large = img.resize((img.width * scale, img.height * scale), Image.Resampling.NEAREST)
                img = img_large.resize(target_size, Image.Resampling.LANCZOS)

        # Save if output path is provided
        if output_path:
            if output_path.lower().endswith('.png'):
                # PNG for pixel-perfect output
                img.save(output_path, optimize=True)

        return img

    def edit_pixel(self, matrix, x, y, new_value):
        """
        Edit a single pixel in the matrix.
        
        Args:
            matrix (numpy.ndarray): Matrix to edit
            x (int): X coordinate (column)
            y (int): Y coordinate (row)
            new_value (int): New value (1-5)
            
        Returns:
            numpy.ndarray: Updated matrix
        """
        if not (0 <= x < self.full_matrix_size[1] and 0 <= y < self.full_matrix_size[0]):
            raise ValueError("Coordinates out of bounds")
        if not (1 <= new_value <= 5):
            raise ValueError("Value must be between 1 and 5")

        matrix[y, x] = new_value  # Note: y, x order for numpy arrays
        return matrix

    def get_panel(self, matrix, panel_row, panel_col):
        """
        Extract a specific panel from the matrix.
        
        Args:
            matrix (numpy.ndarray): Source matrix
            panel_row (int): Panel row index (0-based)
            panel_col (int): Panel column index (0-based)
            
        Returns:
            numpy.ndarray: Panel data as a matrix
        """
        if not (0 <= panel_row < self.layout[0] and 0 <= panel_col < self.layout[1]):
            raise ValueError("Panel coordinates out of bounds")

        start_row = panel_row * self.panel_size
        start_col = panel_col * self.panel_size
        end_row = start_row + self.panel_size
        end_col = start_col + self.panel_size

        return matrix[start_row:end_row, start_col:end_col].copy()

    def set_panel(self, matrix, panel_row, panel_col, panel_data):
        """
        Set a specific panel in the matrix.
        
        Args:
            matrix (numpy.ndarray): Target matrix
            panel_row (int): Panel row index (0-based)
            panel_col (int): Panel column index (0-based)
            panel_data (numpy.ndarray): Panel data to insert
            
        Returns:
            numpy.ndarray: Updated matrix
        """
        if panel_data.shape != (self.panel_size, self.panel_size):
            raise ValueError(f"Panel data must be {self.panel_size}x{self.panel_size}")

        start_row = panel_row * self.panel_size
        start_col = panel_col * self.panel_size
        end_row = start_row + self.panel_size
        end_col = start_col + self.panel_size

        matrix[start_row:end_row, start_col:end_col] = panel_data
        return matrix

    def save_panels(self, matrix, base_filename, target_panel_size=None):
        """
        Save each panel as a separate image file.

        Args:
            matrix (numpy.ndarray): The full image matrix
            base_filename (str): Base filename to use (e.g., 'panel' will create 'panel_1.png', etc.)
            target_panel_size (int, optional): Optional size for each panel image
        """
        base_name, ext = os.path.splitext(base_filename)
        if not ext:  # If no extension provided, use PNG
            ext = '.png'

        panel_number = 1
        for row in range(self.layout[0]):
            for col in range(self.layout[1]):
                # Extract panel
                panel = self.get_panel(matrix, row, col)

                # Convert panel directly to image
                img_array = np.zeros((self.panel_size, self.panel_size), dtype=np.uint8)
                for i in range(self.panel_size):
                    for j in range(self.panel_size):
                        value = panel[i, j]
                        if value in self.color_map:
                            img_array[i, j] = self.color_map[value]

                # Create image from panel array
                panel_img = Image.fromarray(img_array)

                # Resize if target size is provided
                if target_panel_size:
                    if target_panel_size <= self.panel_size * 2:
                        panel_img = panel_img.resize((target_panel_size, target_panel_size),
                                                     Image.Resampling.NEAREST)
                    else:
                        # For large sizes, use high-quality scaling
                        scale = 8
                        img_large = panel_img.resize(
                            (panel_img.width * scale, panel_img.height * scale),
                            Image.Resampling.NEAREST
                        )
                        panel_img = img_large.resize(
                            (target_panel_size, target_panel_size),
                            Image.Resampling.LANCZOS
                        )

                # Save panel with numbered filename
                filename = f"{base_name}_{panel_number}{ext}"
                panel_img.save(filename, optimize=True)
                panel_number += 1

    def apply_dithering(self, matrix, mask_path=None):
        """
        Apply checkerboard dithering to create smooth transitions between shades.
        Creates a gradient effect by alternating pixels in a checkerboard pattern.
        Only applies dithering in regions marked as black in the mask image.

        Args:
            matrix (numpy.ndarray): Input numpy matrix with values 1-5
            mask_path (str, optional): Path to mask image (black allows dithering, white prevents it)

        Returns:
            numpy.ndarray: Dithered numpy matrix with values 1-5
        """
        height, width = matrix.shape
        result = matrix.copy()

        # Load and process mask if provided
        if mask_path:
            with Image.open(mask_path) as mask_img:
                mask_img = mask_img.convert('L')
                mask_img = mask_img.resize((width, height), Image.Resampling.NEAREST)
                mask = np.array(mask_img) < 128  # True where dithering is allowed
        else:
            mask = np.ones_like(matrix, dtype=bool)

        def get_lighter_darker_shades(value):
            """Get the next lighter and darker shades if they exist."""
            if value < 1 or value > 5:
                return None, None
            lighter = value - 1 if value > 1 else None
            darker = value + 1 if value < 5 else None
            return lighter, darker

        changes = 0
        # Apply checkerboard dithering
        for y in range(height):
            for x in range(width):
                if not mask[y, x]:
                    continue

                current_value = matrix[y, x]
                lighter, darker = get_lighter_darker_shades(current_value)

                # Get the values of surrounding pixels
                neighbors = []
                if x > 0:
                    neighbors.append(matrix[y, x - 1])
                if x < width - 1:
                    neighbors.append(matrix[y, x + 1])
                if y > 0:
                    neighbors.append(matrix[y - 1, x])
                if y < height - 1:
                    neighbors.append(matrix[y + 1, x])

                # Find if we should blend with a lighter or darker shade
                blend_with = None
                if lighter in neighbors and darker not in neighbors:
                    blend_with = lighter
                elif darker in neighbors and lighter not in neighbors:
                    blend_with = darker
                elif lighter in neighbors and darker in neighbors:
                    # If both are present, prefer the more common one
                    neighbor_counts = {n: neighbors.count(n) for n in [lighter, darker]}
                    blend_with = max(neighbor_counts.items(), key=lambda x: x[1])[0]

                # Apply checkerboard pattern if we found a shade to blend with
                if blend_with is not None:
                    is_checker = (x + y) % 2 == 0
                    if is_checker and current_value != blend_with:
                        result[y, x] = blend_with
                        changes += 1

        print(f"Made {changes} dithering changes")
        return result


class MozabrickInstructionExporter:
    """
    Class to export building instructions for the mosaic.
    """

    def __init__(self, processor):
        """
        Initialize the exporter with a processor.
        
        Args:
            processor (MozabrickProcessor): The processor to use
        """
        self.processor = processor
        self.current_matrix = None

    def _encode_row(self, row):
        """Encode a single row into runs of values with positions."""
        runs = []
        current_val = row[0]
        start_pos = 0
        count = 1

        for i, val in enumerate(row[1:], 1):
            if val == current_val:
                count += 1
            else:
                runs.append((current_val, start_pos, count))
                current_val = val
                start_pos = i
                count = 1
        runs.append((current_val, start_pos, count))
        return runs

    def _generate_header_row(self):
        """Generate properly aligned column numbers."""
        header = ["  "]  # Two leading spaces
        for i in range(1, self.processor.panel_size + 1):
            if i < 10:
                header.append(f" {i}")  # Extra space for single digits
            else:
                header.append(str(i))
        return " ".join(header)

    def export_text(self, matrix, output_path):
        """
        Export instructions as text file showing shade values.
        
        Args:
            matrix (numpy.ndarray): Matrix to export
            output_path (str): Path to save the text file
        """
        header_row = self._generate_header_row()

        with open(output_path, 'w', encoding='utf-8') as f:
            panel_number = 1
            for panel_row in range(self.processor.layout[0]):
                for panel_col in range(self.processor.layout[1]):
                    f.write(f"\nPanel {panel_number}:\n")
                    f.write(header_row + '\n')
                    panel = self.processor.get_panel(matrix, panel_row, panel_col)

                    for row_idx, row in enumerate(panel, 1):
                        f.write(f"{row_idx:2d} ")
                        values = [" " * 2] * self.processor.panel_size

                        for col_idx, val in enumerate(row):
                            values[col_idx] = f" {val}" if val < 10 else str(val)

                        f.write(" ".join(values))
                        f.write(f" {row_idx}\n")

                    panel_number += 1
                    if panel_col == self.processor.layout[1] - 1 and panel_row < self.processor.layout[0] - 1:
                        f.write('\n' + '-' * 80 + '\n')

    def export_image(self, matrix, output_path):
        """
        Export instructions as image with symbols.
        
        Args:
            matrix (numpy.ndarray): Matrix to export
            output_path (str): Path to save the image
        """
        self.current_matrix = matrix

        # Base dimensions
        margin = 50
        symbol_size = 20
        spacing = 5
        row_height = symbol_size + spacing
        col_width = symbol_size + spacing

        # Calculate pattern area
        pattern_width = self.processor.panel_size * col_width + margin * 2
        pattern_height = self.processor.panel_size * row_height + margin * 2

        # Calculate total dimensions including legend area
        legend_width = 200  # Fixed width for legend
        total_width = (pattern_width * self.processor.layout[1]) + legend_width + margin
        total_height = pattern_height * self.processor.layout[0]

        # Create surface
        surface = cairo.ImageSurface(cairo.FORMAT_RGB24, int(total_width), int(total_height))
        ctx = cairo.Context(surface)

        # White background
        ctx.set_source_rgb(1, 1, 1)
        ctx.paint()

        # Draw panels
        panel_number = 1
        for panel_row in range(self.processor.layout[0]):
            for panel_col in range(self.processor.layout[1]):
                base_x = panel_col * pattern_width + margin
                base_y = panel_row * pattern_height + margin
                self._draw_panel(ctx, matrix, panel_row, panel_col,
                                 panel_number, pattern_width, pattern_height,
                                 base_x, base_y, symbol_size, spacing)
                panel_number += 1

        # Draw legend in the right margin
        legend_x = pattern_width * self.processor.layout[1] + margin
        legend_y = margin * 2
        self._draw_legend(ctx, legend_x, legend_y, symbol_size)

        surface.write_to_png(output_path)

    def _draw_panel(self, ctx, matrix, panel_row, panel_col, panel_number,
                    panel_width, panel_height, base_x, base_y, symbol_size, spacing):
        """Draw a single panel."""
        panel = self.processor.get_panel(matrix, panel_row, panel_col)

        # Draw panel header
        ctx.set_source_rgb(0, 0, 0)
        ctx.select_font_face("Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(24)
        ctx.move_to(base_x, base_y - 10)
        ctx.show_text(f"Panel {panel_number}")

        # Draw column numbers
        self._draw_column_numbers(ctx, base_x, base_y + symbol_size, symbol_size, spacing)

        # Draw pattern
        for row_idx, row in enumerate(panel):
            y = base_y + (row_idx + 2) * (symbol_size + spacing)

            # Row numbers (plain text)
            self._draw_text(ctx, base_x - symbol_size - spacing, y, row_idx + 1, symbol_size)
            self._draw_text(ctx, base_x + self.processor.panel_size * (symbol_size + spacing) + spacing,
                            y, row_idx + 1, symbol_size)

            # Draw symbols with sequence numbers
            for value, start_pos, length in self._encode_row(row):
                x = base_x + start_pos * (symbol_size + spacing)
                for i in range(length):
                    self._draw_symbol(ctx, x, y, i + 1, symbol_size, value)
                    x += symbol_size + spacing

    def _draw_column_numbers(self, ctx, base_x, base_y, symbol_size, spacing):
        """Draw column numbers."""
        ctx.select_font_face("Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_source_rgb(0, 0, 0)
        ctx.set_font_size(symbol_size * 0.4)

        for i in range(self.processor.panel_size):
            x = base_x + i * (symbol_size + spacing)
            number = str(i + 1)
            extents = ctx.text_extents(number)
            ctx.move_to(x + (symbol_size - extents.width) / 2, base_y - symbol_size)
            ctx.show_text(number)

    def _draw_text(self, ctx, x, y, number, size):
        """Draw plain text number."""
        ctx.select_font_face("Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        ctx.set_source_rgb(0, 0, 0)
        ctx.set_font_size(size * 0.4)
        text = str(number)
        extents = ctx.text_extents(text)
        ctx.move_to(x + (size - extents.width) / 2, y - size / 2 + extents.height / 2)
        ctx.show_text(text)

    def _draw_number(self, ctx, x, y, size, number):
        """Draw number in top-left corner."""
        if number is None:
            return

        ctx.select_font_face("Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_NORMAL)
        font_size = size * 0.4
        ctx.set_font_size(font_size)

        ctx.set_source_rgb(0, 0, 0)

        # Position in top-left with small padding
        padding = size * 0.1
        text = str(number)
        ctx.move_to(x + padding, y - size + font_size + padding)
        ctx.show_text(text)

    def _draw_open_right_triangle(self, ctx, x, y, size):
        """Draw shape 3: open right triangle (L shape)."""
        ctx.set_source_rgb(0.3, 0.3, 0.3)  # Dark gray
        ctx.set_line_width(1)
        # Draw just two sides making an L shape
        ctx.move_to(x + size, y)
        ctx.line_to(x + size, y - size)
        ctx.line_to(x, y)
        ctx.stroke()

    def _draw_closed_triangle(self, ctx, x, y, size):
        """Draw shape 4: closed triangle outline."""
        ctx.set_source_rgb(0, 0, 0)  # Black
        ctx.set_line_width(1)
        # Draw all three sides
        ctx.move_to(x, y)
        ctx.line_to(x + size, y)
        ctx.line_to(x + size, y - size)
        ctx.line_to(x, y)
        ctx.stroke()

    def _draw_symbol(self, ctx, x, y, number, size, shade):
        """Draw symbol with number."""
        if shade == 1:  # Light gray filled square (no border)
            self._draw_light_square(ctx, x, y, size)
        elif shade == 2:  # Empty square with dark gray outline
            self._draw_outline_square(ctx, x, y, size)
        elif shade == 3:  # Open right triangle (L shape)
            self._draw_open_right_triangle(ctx, x, y, size)
        elif shade == 4:  # Closed triangle outline
            self._draw_closed_triangle(ctx, x, y, size)
        elif shade == 5:  # Black filled triangle
            self._draw_filled_triangle(ctx, x, y, size)

        self._draw_number(ctx, x, y, size, number)

    def _draw_light_square(self, ctx, x, y, size):
        """Draw light gray filled square without border."""
        ctx.rectangle(x, y - size, size, size)
        ctx.set_source_rgb(0.9, 0.9, 0.9)  # Light gray
        ctx.fill()

    def _draw_outline_square(self, ctx, x, y, size):
        """Draw square outline in darker gray."""
        ctx.rectangle(x, y - size, size, size)
        ctx.set_source_rgb(0.6, 0.6, 0.6)  # Medium gray
        ctx.set_line_width(1)
        ctx.stroke()

    def _draw_filled_triangle(self, ctx, x, y, size):
        """Draw black filled triangle."""
        # Save the context state
        ctx.save()

        # Draw the triangle
        ctx.move_to(x, y)
        ctx.line_to(x + size, y)
        ctx.line_to(x + size, y - size)
        ctx.close_path()
        ctx.set_source_rgb(0, 0, 0)
        ctx.fill()

        # Restore context before number drawing
        ctx.restore()

    def _count_shapes(self, matrix):
        """Count total occurrences of each shape value in the matrix."""
        import numpy as np
        counts = {}
        for i in range(1, 6):  # Count shapes 1 through 5
            counts[i] = np.count_nonzero(matrix == i)
        return counts

    def _draw_legend(self, ctx, x, y, size):
        """Draw legend with actual shape counts."""
        items = [(1, "Light gray"),
                 (2, "Outline"),
                 (3, "Open triangle"),
                 (4, "Closed triangle"),
                 (5, "Black")]

        # Get actual counts
        counts = self._count_shapes(self.current_matrix)

        spacing = size * 2
        ctx.select_font_face("Arial", cairo.FONT_SLANT_NORMAL, cairo.FONT_WEIGHT_BOLD)
        ctx.set_font_size(size * 0.7)

        for i, (value, label) in enumerate(items):
            y_pos = y + i * spacing
            self._draw_symbol(ctx, x, y_pos, None, size, value)
            ctx.set_source_rgb(0, 0, 0)
            ctx.move_to(x + size * 1.5, y_pos - size / 2)
            ctx.show_text(f"{label}: {counts[value]}")