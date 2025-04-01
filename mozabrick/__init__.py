"""
MozaBrick - A tool for creating pixel art mosaics that can be built with physical bricks.

This package provides classes for processing images into pixelated mosaics with
5 different shading levels and generating assembly instructions.
"""

from .mozabrick import MozabrickProcessor, MozabrickInstructionExporter

__version__ = '0.1.0'
__all__ = ['MozabrickProcessor', 'MozabrickInstructionExporter']
