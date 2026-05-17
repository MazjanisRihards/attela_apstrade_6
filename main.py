import requests, io, time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

headers = {"User-Agent": "Mozilla/5.0"}

url1 = "https://picsum.photos/id/1060/500/400"
url2 = "https://picsum.photos/id/292/500/400"
url3 = "https://picsum.photos/id/1080/500/400"
url4 = "https://picsum.photos/id/164/500/400"
url5 = "https://picsum.photos/id/145/500/400"


def load_image(url, retries=5, delay=3):
    for attempt in range(retries):
        r = requests.get(url, headers=headers, timeout=30)
        if r.status_code == 429:
            print(f"429 - gaidu {delay}s... (meginjums {attempt+1}/{retries})")
            time.sleep(delay)
            delay *= 2
            continue
        r.raise_for_status()
        return np.array(Image.open(io.BytesIO(r.content)).convert("RGB"))
    raise Exception(f"Neizdevas ieladet: {url}")


def add_gaussian_noise(img, sigma=25):
    noisy = img.astype(np.float64) + np.random.normal(0, sigma, img.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper(img, amount=0.05):
    noisy = img.copy()
    h, w = img.shape[:2]
    n = int(amount * h * w)
    noisy[np.random.randint(0, h, n), np.random.randint(0, w, n)] = 255
    noisy[np.random.randint(0, h, n), np.random.randint(0, w, n)] = 0
    return noisy


def gauss_kernel(k=5, sigma=1.0):
    ax = np.arange(-(k // 2), k // 2 + 1, dtype=np.float64)
    g = np.exp(-ax**2 / (2 * sigma**2))
    kernel = np.outer(g, g)
    return kernel / kernel.sum()


def gaussian_filter(img, k=5, sigma=1.0):
    kernel = gauss_kernel(k, sigma)
    pad = k // 2
    out = np.zeros_like(img, dtype=np.float64)
    for c in range(3):
        ch = np.pad(img[:, :, c], pad, mode="edge").astype(np.float64)
        windows = np.lib.stride_tricks.sliding_window_view(ch, (k, k))
        out[:, :, c] = np.sum(windows * kernel, axis=(-2, -1))
    return np.clip(out, 0, 255).astype(np.uint8)


def median_filter(img, k=3):
    pad = k // 2
    out = np.zeros_like(img)
    for c in range(3):
        ch = np.pad(img[:, :, c], pad, mode="edge").astype(np.float64)
        windows = np.lib.stride_tricks.sliding_window_view(ch, (k, k))
        out[:, :, c] = np.median(windows, axis=(-2, -1))
    return out.astype(np.uint8)


def detect_noise(img):
    gray = img.mean(axis=2)
    h, w = gray.shape

    pad = np.pad(gray, 1, mode='edge')

    neighbors = np.stack([
        pad[0:h,   0:w],   pad[0:h,   1:w+1], pad[0:h,   2:w+2],
        pad[1:h+1, 0:w],                       pad[1:h+1, 2:w+2],
        pad[2:h+2, 0:w],   pad[2:h+2, 1:w+1], pad[2:h+2, 2:w+2]
    ])

    neighbor_max = neighbors.max(axis=0)
    neighbor_min = neighbors.min(axis=0)

    isolated_salt    = ((gray >= 250) & (neighbor_max < 200)).mean()
    isolated_pepper  = ((gray <= 5)   & (neighbor_min > 55)).mean()

    return "salt_pepper" if (isolated_salt + isolated_pepper) > 0.005 else "gaussian"

def combined(img, noise_type, gauss_k=5, sigma=1.0, median_k=3):
    if noise_type == "salt_pepper":
        return gaussian_filter(median_filter(img, k=median_k), k=gauss_k, sigma=sigma)
    else:
        return median_filter(gaussian_filter(img, k=gauss_k, sigma=sigma), k=median_k)


def psnr(orig, proc):
    mse = np.mean((orig.astype(float) - proc.astype(float)) ** 2)
    return float("inf") if mse == 0 else 20 * np.log10(255.0 / np.sqrt(mse))


def show(img, noisy, title, gauss_k=5, sigma=1.0, median_k=3):
    noise_type = detect_noise(noisy)
    gauss_result = gaussian_filter(noisy, k=gauss_k, sigma=sigma)
    median_result = median_filter(noisy, k=median_k)
    combined_result = combined(noisy, noise_type, gauss_k, sigma, median_k)

    p_noisy    = psnr(img, noisy)
    p_gauss    = psnr(img, gauss_result)
    p_median   = psnr(img, median_result)
    p_combined = psnr(img, combined_result)

    fig, axes = plt.subplots(1, 5, figsize=(24, 4))
    fig.suptitle(title, fontsize=13)

    for ax, data, label in zip(axes,
        [img, noisy, gauss_result, median_result, combined_result],
        [
            "Oriģināls",
            f"Ar troksni\nPSNR: {p_noisy:.1f} dB",
            f"Gausa filtrs (σ={sigma})\nPSNR: {p_gauss:.1f} dB",
            f"Mediānas filtrs (k={median_k})\nPSNR: {p_median:.1f} dB",
            f"Kombinēts ({noise_type})\nPSNR: {p_combined:.1f} dB",
        ]):
        ax.imshow(data)
        ax.set_title(label, fontsize=9)
        ax.axis("off")

    plt.tight_layout()
    plt.savefig(title.replace(" ", "_") + ".png", dpi=120)
    plt.show()


np.random.seed(42)

img1 = load_image(url1)
time.sleep(2)
img2 = load_image(url2)
time.sleep(2)
img3 = load_image(url3)
time.sleep(2)
img4 = load_image(url4)
time.sleep(2)
img5 = load_image(url5)

show(img1, add_gaussian_noise(img1, sigma=25), "Attels 1 - Kefejnica",     gauss_k=5, sigma=1.2, median_k=3)
show(img2, add_salt_pepper(img2, amount=0.06), "Attels 2 - Darzeni",     gauss_k=5, sigma=1.0, median_k=3)
show(img3, add_gaussian_noise(img3, sigma=20), "Attels 3 - Zemenes",      gauss_k=5, sigma=0.8, median_k=3)
show(img4, add_salt_pepper(img4, amount=0.08), "Attels 4 - Pilseta", gauss_k=5, sigma=1.0, median_k=5)
show(img5, add_gaussian_noise(img5, sigma=30), "Attels 5 - Gitara",        gauss_k=7, sigma=1.5, median_k=3)