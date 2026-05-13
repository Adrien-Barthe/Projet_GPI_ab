##############################
#IMPORT
##############################

import numpy as np
import matplotlib.pyplot as plt
import cv2
import tifffile as tif
from scipy import ndimage

from PIL import Image
import importlib.util
import subprocess
import sys
import requests
from io import BytesIO  # Ensure this import is present

from skimage import io, transform
from scipy.signal import correlate2d

import urllib.request
#import io
from Bio.PDB import PDBParser, PDBIO
from mpl_toolkits.mplot3d import Axes3D


# Ensure scikit-image is available for bilateral denoising.
if importlib.util.find_spec('skimage') is None:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'scikit-image'])

from skimage.restoration import denoise_bilateral


# Install required package (run once)
# %pip install mrcfile

import mrcfile

# %pip install gdown

import gdown
import mrcfile
import os

# if needed, you can install packages using pip
# %pip install scikit-image

# Set random seed for reproducibility
np.random.seed(42)


#func

def bin_image(image, bin_size):
    """
    Performs binning (spatial downsampling) on an image.

    Args:
        image: Input image as numpy array
        bin_size: Size of the bin (e.g., 2 for 2x2 binning)

    Returns:
        Binned image with reduced resolution
    """
    h, w = image.shape
    # Calculate new dimensions
    new_h = h // bin_size
    new_w = w // bin_size

    # Reshape and compute mean of each bin
    binned = image[:new_h*bin_size, :new_w*bin_size].reshape(new_h, bin_size, new_w, bin_size)
    binned = binned.mean(axis=(1, 3)).astype(np.uint8)

    return binned


#code




file_id = "1Qj30jSXcHEpkzE04cisbP6ljtnQ2Ausr"
output = "14sep05c_00024sq_00006hl_00003es_c.mrc"

print("Downloading MRC file...")
gdown.download(id=file_id, output=output, quiet=False)


file_size = os.path.getsize(output)
print(f"\nFile size: {file_size / (1024**2):.2f} MB")

with open(output, "rb") as f:
    header = f.read(4)

print("First 4 bytes:", header)

if header.startswith(b"<"):
    raise ValueError("❌ Downloaded file is HTML, not MRC. Check sharing permissions.")

if file_size < 10000:
    raise ValueError("❌ File too small — likely incorrect download.")

print("✓ File appears valid")

print("\nLoading MRC file...")

with mrcfile.open(output, permissive=True) as mrc:
    if mrc.data is None:
        raise ValueError("❌ MRC file contains no readable data")
    data = mrc.data.copy()

print("✓ MRC loaded successfully")

print("\n" + "="*60)
print("MRC DATA INFO")
print("="*60)
print("Shape:", data.shape)
print("Dtype:", data.dtype)
print("Min/Max:", data.min(), data.max())


if data.ndim == 3:
    print("\nDetected: 3D volume")

    z, y, x = np.array(data.shape) // 2

    # ------------------------------------------
    # CENTRAL SLICES (RAW)
    # ------------------------------------------
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    slices = [
        (data[z, :, :], "XY slice"),
        (data[:, y, :], "XZ slice"),
        (data[:, :, x], "YZ slice"),
    ]

    for ax, (img, title) in zip(axes, slices):
        vmin = np.percentile(img, 5)
        vmax = np.percentile(img, 95)
        ax.imshow(img, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

    print("\nGenerating summed projections...")

    proj_xy = np.sum(data, axis=0)
    proj_xz = np.sum(data, axis=1)
    proj_yz = np.sum(data, axis=2)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    projections = [
        (proj_xy, "XY Projection (sum Z)"),
        (proj_xz, "XZ Projection (sum Y)"),
        (proj_yz, "YZ Projection (sum X)")
    ]

    for ax, (proj, title) in zip(axes, projections):
        vmin = np.percentile(proj, 5)
        vmax = np.percentile(proj, 95)
        ax.imshow(proj, cmap='gray', vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis('off')

    plt.tight_layout()
    plt.show()

elif data.ndim == 2:
    print("\nDetected: single 2D image")

    bin_factor = 4
    print(f"Application du binning (facteur {bin_factor}x{bin_factor})...")

    # On applique la fonction sur les données extraites
    data_binned = bin_image(data, bin_factor)



    print("Normalisation de l'image...")
    # On convertit en float32 (pour la précision), on divise par le max et on centre sur zéro
    data_normalized = data_binned.astype(np.float32) / data_binned.max()
    data_normalized = data_normalized - data_normalized.mean()

    vmin = np.percentile(data_normalized, 5)
    vmax = np.percentile(data_normalized, 95)

    plt.imshow(data_normalized, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title(f"Micrographie (Binned {bin_factor}x & Normalisée)")
    plt.axis('off')
    plt.show()

    print(f"Mean: {data_binned.mean():.2f}, Min: {data_binned.min():.2f}, Max: {data_binned.max():.2f}")
    print(f"Mean: {data_normalized.mean():.6f}, Min: {data_normalized.min():.2f}, Max: {data_normalized.max():.2f}")



else:
    print("\n⚠️ Unrecognized format")




with mrcfile.open("14sep05c_00024sq_00006hl_00003es_c.mrc", permissive=True) as mrc:
    h = mrc.header

    print("=" * 60)
    print("MRC HEADER (FORMATTED SUMMARY)")
    print("=" * 60)

    print(f"Dimensions (nx, ny, nz): {h.nx}, {h.ny}, {h.nz}")
    print(f"Mode (data type): {h.mode}")

    print(f"Start (nxstart, nystart, nzstart): {h.nxstart}, {h.nystart}, {h.nzstart}")
    print(f"Grid size (mx, my, mz): {h.mx}, {h.my}, {h.mz}")

    print(f"Cell dimensions (Å): {h.cella.x}, {h.cella.y}, {h.cella.z}")
    print(f"Cell angles (°): {h.cellb.alpha}, {h.cellb.beta}, {h.cellb.gamma}")

    print(f"Axis mapping (mapc, mapr, maps): {h.mapc}, {h.mapr}, {h.maps}")

    print(f"Density stats: min={h.dmin}, max={h.dmax}, mean={h.dmean}")

    print(f"Space group: {h.ispg}")
    print(f"Extended header size: {h.nsymbt}")



