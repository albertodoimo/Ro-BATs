import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os
import numpy as np

# Parameters
marker_size = 100  # pixels     # pixels
markers_per_page = 40
total_markers = 40

# A4 size in inches at 300 dpi
dpi = 300
a4_width_in = 8.27
a4_height_in = 11.69

# ArUco dictionary
#dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
dictionary = aruco.extendDictionary(40, 3)

def main():
    marker_size = 3
    num_markers = 100
    max_correction_bits = 3

    # Initialize an empty custom dictionary
    dictionary.maxCorrectionBits = max_correction_bits

    for i in range(num_markers):
        marker_bits = np.random.randint(0, 2, (marker_size, marker_size), dtype=np.uint8)
        print(marker_bits)

        # Convert bits to compressed byte format
        marker_compressed = cv2.aruco.Dictionary.getByteListFromBits(marker_bits)

        # Add the marker as a new row in bytesList
        dictionary.bytesList = np.append(dictionary.bytesList, marker_compressed, axis=0)

    print(f"✅ Custom dictionary created with {num_markers} markers of {marker_size}x{marker_size} bits.")

if __name__ == "__main__":
    main()

# PDF setup
pdf_filename = "aruco_markers_a4.pdf"
pdf = PdfPages(pdf_filename)

# Layout: max 5 columns x 5 rows per page
cols = 5
rows = 8

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
        ax.set_title(f'ID: {marker_id}', fontsize=8,pad=0.1)
        ax.axis('off')

    # Hide unused subplots
    for j in range(i + 1, rows * cols):
        row = j // cols
        col = j % cols
        axs[row][col].axis('off')

    plt.tight_layout(pad=1.3)
    pdf.savefig(fig)
    plt.close(fig)

# Save PDF
pdf.close()
print(f"PDF with {total_markers} ArUco markers saved as '{pdf_filename}' (A4 layout)")


