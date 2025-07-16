import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import math

# Parameters
marker_size = 220  # pixels
spacing = 10       # pixels
markers_per_page = 15
total_markers = 30

# A4 size in inches at 100 dpi
dpi = 100
a4_width_in = 8.27
a4_height_in = 11.69

# ArUco dictionary
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# PDF setup
pdf_filename = "aruco_markers_a4.pdf"
pdf = PdfPages(pdf_filename)

# Layout: max 3 columns x 5 rows per page
cols = 3
rows = 5

for page_start in range(0, total_markers, markers_per_page):
    fig, axs = plt.subplots(rows, cols, figsize=(a4_width_in, a4_height_in), dpi=dpi)

    for i in range(markers_per_page):
        marker_id = page_start + i
        if marker_id >= total_markers:
            break

        img = aruco.generateImageMarker(dictionary, marker_id, marker_size)

        row = i // cols
        col = i % cols

        ax = axs[row][col]
        ax.imshow(img, cmap='gray')
        ax.set_title(f'ID: {marker_id}', fontsize=8)
        ax.axis('off')

    # Hide unused subplots
    for j in range(i + 1, rows * cols):
        row = j // cols
        col = j % cols
        axs[row][col].axis('off')

    plt.tight_layout(pad=1.5)
    pdf.savefig(fig)
    plt.close(fig)

# Save PDF
pdf.close()
print(f"PDF with {total_markers} ArUco markers saved as '{pdf_filename}' (A4 layout)")
