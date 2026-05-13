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


def apply_padding(img, pad_width, mode='constant', constant_value=0):
    """
    Apply padding to an image using NumPy.

    Parameters:
    -----------
    img : ndarray
        Input image
    pad_width : int or tuple
        Number of pixels to pad on each side
    mode : str
        Padding mode: 'constant', 'edge', 'reflect', 'wrap'
    constant_value : int/float
        Value used for constant padding
    """
    return np.pad(img, pad_width, mode=mode, constant_values=constant_value)


def get_pdb_coords(pdb_file):
    """
    Extract atomic coordinates from a PDB file.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("molecule", pdb_file)
    coords = np.array([atom.get_coord() for atom in structure.get_atoms()])

    # Center the coordinates
    coords -= np.mean(coords, axis=0)
    return coords

padding = 5

def create_projection(coord1, coord2, bins=256):
    # Determine bounds
    min1, max1 = coord1.min() - padding, coord1.max() + padding
    min2, max2 = coord2.min() - padding, coord2.max() + padding

    # 2D histogram (counts = density)
    H, xedges, yedges = np.histogram2d(coord1, coord2, bins=bins,
                                       range=[[min1, max1], [min2, max2]])

    # Normalize for display
    H = H / H.max()

    return H


def template_matching_ncc(image, template):
    """
    Perform template matching with padding, inspired by the notebook.
    """
    tmp_h, tmp_w = template.shape
    pad_h, pad_w = tmp_h // 2, tmp_w // 2

    # 1. Pad image (Le notebook gère le padding ici)
    padded = np.pad(image,
                    ((pad_h, pad_h), (pad_w, pad_w)),
                    mode='constant',
                    constant_values=0)

    # 2. Perform matching (NCC)
    # 'valid' signifie qu'on calcule uniquement là où le template rentre entièrement (grâce au padding)
    result = correlate2d(padded, template, mode='valid')

    # Normalisation du résultat entre -1 et 1 (Approximation rapide du vrai NCC)
    if result.max() != 0:
        result = result / result.max()

    return result


#main




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
    data_binned = bin_image(data, bin_factor)

    # Normalisation
    data_normalized = data_binned.astype(np.float32) / data_binned.max()
    data_normalized = data_normalized - data_normalized.mean()

    # PADDING
    pad_size = 50
    print(f"Application du padding (taille={pad_size})...")

    data_padded = apply_padding(data_normalized, pad_size, mode='constant', constant_value=0)

    # plot
    vmin = np.percentile(data_padded, 5)
    vmax = np.percentile(data_padded, 95)
    plt.imshow(data_padded, cmap='gray', vmin=vmin, vmax=vmax)
    plt.title(f"Image binnée, normalisée et paddée (pad={pad_size})")
    plt.axis('off')
    plt.show()

    print(f"Dimensions après padding : {data_padded.shape}")

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

# --- PREPARE THE TEMPLATE ---
original_pixel_size = 0.66
bin_factor = 4
effective_pixel_size = original_pixel_size * bin_factor

# 1. Obtenir et centrer les coordonnées (comme dans le notebook)
coords = get_pdb_coords("6BDF.pdb")
coords_centered = coords - coords.mean(axis=0)

# 2. Calculer le bon nombre de 'bins' (pixels) pour respecter l'échelle
# On calcule la largeur physique totale (en Å) divisée par la taille d'un pixel binné
largeur_physique_A = (coords_centered[:, 0].max() + padding) - (coords_centered[:, 0].min() - padding)
nombre_de_pixels = int(largeur_physique_A / effective_pixel_size)

print(f"Création d'un template de {nombre_de_pixels}x{nombre_de_pixels} pixels...")

# 3. Générer la projection (XY = Top view)
# On passe x (coord[:,0]) et y (coord[:,1])
template = create_projection(coords_centered[:, 0], coords_centered[:, 1], bins=nombre_de_pixels)

# Le notebook utilise la transposée (.T) pour l'affichage, on le fait ici pour l'image finale
template = template.T

# 4. Préparation finale pour le NCC
# La fonction du notebook a déjà fait H = H / H.max() (entre 0 et 1)
# Il ne reste plus qu'à centrer sur zéro (Zero-mean) et inverser les couleurs
template = template - template.mean()
template = -template

# Affichage du template final
plt.imshow(template, cmap='gray', origin='lower')
plt.title("Template 2D (Via fonction du Notebook)")
plt.axis('off')
plt.show()


# --- 1. TEMPLATE MATCHING (NCC) ---
print("\nExécution du Template Matching...")
match_result = template_matching_ncc(data_normalized, template)

print("Statistiques de la carte NCC :")
print(f"Max value: {match_result.max():.4f}")
print(f"Mean value: {match_result.mean():.4f}")

# --- 2. OPTIMIZE THE THRESHOLD ---
# Stratégie "Relative threshold" : par exemple 75% du maximum
seuil_relatif = 0.30
print(f"Application d'un seuil relatif à {seuil_relatif*100}% du maximum...")

# --- 3. EXTRACT THE PARTICLES ---
from skimage.feature import peak_local_max
import matplotlib.patches as patches

# On cherche les pics locaux qui dépassent notre seuil
# min_distance empêche de trouver 2 pics sur la même particule
distance_min = template.shape[0] // 2
coords = peak_local_max(match_result, min_distance=distance_min, threshold_abs=seuil_relatif)

print(f"🎉 {len(coords)} particules extraites !")

# --- 4. AFFICHAGE DES RÉSULTATS ---
fig, axes = plt.subplots(1, 2, figsize=(15, 7))

# Carte de corrélation
im0 = axes[0].imshow(match_result, cmap='hot', vmin=0, vmax=1)
axes[0].set_title('NCC Map (Hot = Match)')
plt.colorbar(im0, ax=axes[0])

# Image originale avec les particules extraites
axes[1].imshow(data_binned, cmap='gray')
axes[1].set_title(f"Particules extraites (Seuil: {seuil_relatif})")

for y, x in coords:
    # On dessine une boîte (box size) de la taille du template
    rect = patches.Rectangle((x - template.shape[1]//2, y - template.shape[0]//2),
                             template.shape[1], template.shape[0],
                             linewidth=1.5, edgecolor='lime', facecolor='none')
    axes[1].add_patch(rect)

plt.tight_layout()
plt.show()
