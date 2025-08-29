import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.figure import Figure

os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Configuration
marker_ids = [12, 13, 14]
dictionary = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)

# Page size and DPI
dpi = 300
a4_width_in, a4_height_in = 8.27, 11.69  # A4 in inches
margin_in = 0.5  # Top, bottom, left, right margin in inches

# Determine maximum square size that fits in the page with margins
max_square_in = min(a4_width_in, a4_height_in) - 2 * margin_in
marker_img_size = 1000  # Pixels for generating high-res marker

# Create PDF
with PdfPages("arena_markers.pdf") as pdf:
    for marker_id in marker_ids:
        # Generate marker image
        marker_img = cv2.aruco.generateImageMarker(dictionary, marker_id, marker_img_size)

        # Create figure sized to A4
        fig = Figure(figsize=(a4_width_in, a4_height_in), dpi=dpi)
        ax = fig.subplots()

        # Remove axes
        ax.axis('off')

        # Add marker ID as a title in top margin
        fig.suptitle(f"Aruco Marker ID: {marker_id}", fontsize=14, y=0.80)

        # Compute square marker placement coordinates in inches
        x0 = (a4_width_in - max_square_in) / 2
        y0 = (a4_height_in - max_square_in) / 2
        x1 = x0 + max_square_in
        y1 = y0 + max_square_in

        # Plot marker image in the calculated extent
        ax.imshow(marker_img, cmap='gray', interpolation='nearest', extent=[x0, x1, y0, y1])
        ax.set_xlim(0, a4_width_in)
        ax.set_ylim(0, a4_height_in)
        ax.set_aspect('equal')

        # Save to PDF and close figure
        pdf.savefig(fig)
        plt.close(fig)

print("PDF with square ArUco markers and IDs saved as: aruco_markers.pdf")
