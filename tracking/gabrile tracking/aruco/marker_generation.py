import cv2
import cv2.aruco as aruco
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Parameters
marker_size = 220  # in pixels
num_markers = 30
dictionary = aruco.getPredefinedDictionary(aruco.DICT_6X6_250)

# PDF setup
pdf_filename = "aruco_markers_6x6_250.pdf"
pdf = PdfPages(pdf_filename)

# Generate and plot each marker
for marker_id in range(num_markers):
    img = aruco.generateImageMarker(dictionary, marker_id, marker_size)

    fig, ax = plt.subplots(figsize=(2.2, 2.2), dpi=100)  # 220 px @ 100 dpi
    ax.imshow(img, cmap='gray')
    ax.set_title(f'ID: {marker_id}', fontsize=10)
    ax.axis('off')

    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

# Save PDF
pdf.close()
print(f"PDF with {num_markers} ArUco markers saved as '{pdf_filename}'")
