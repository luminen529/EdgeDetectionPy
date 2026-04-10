import os
import csv
import cv2
import numpy as np

INPUT_DIR = "input_images"
OUTPUT_DIR = "output_edge_study"

GAUSSIAN_SIGMAS = [10, 25]  # 가우시안 노이즈 세기
SP_AMOUNTS = [0.02, 0.08]  # Salt & Pepper 노이즈 비율
THRESHOLD_RATIO = 0.30  # 에지 임계값 = 최대값 * 비율

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def normalize_to_uint8(img: np.ndarray) -> np.ndarray:
    img = img.astype(np.float32)
    img = np.abs(img)
    max_val = img.max()
    if max_val < 1e-8:
        return np.zeros_like(img, dtype=np.uint8)
    img = img / max_val * 255.0
    return np.clip(img, 0, 255).astype(np.uint8)

def threshold_edge(mag: np.ndarray, ratio: float = THRESHOLD_RATIO) -> np.ndarray:
    mag = np.abs(mag.astype(np.float32))
    thresh = mag.max() * ratio
    edge = np.zeros_like(mag, dtype=np.uint8)
    edge[mag > thresh] = 255
    return edge

def put_title(img: np.ndarray, title: str) -> np.ndarray:
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    canvas = img.copy()
    cv2.rectangle(canvas, (0, 0), (canvas.shape[1], 24), (255, 255, 255), -1)
    cv2.putText(
        canvas, title, (6, 17),
        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA
    )
    return canvas


def make_contact_sheet(images, cols=3, bg_color=255):
    if not images:
        raise ValueError("images list is empty")

    converted = []
    max_h = 0
    max_w = 0
    for img in images:
        if len(img.shape) == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
        converted.append(img)
        max_h = max(max_h, img.shape[0])
        max_w = max(max_w, img.shape[1])

    padded = []
    for img in converted:
        h, w = img.shape[:2]
        canvas = np.full((max_h, max_w, 3), bg_color, dtype=np.uint8)
        canvas[:h, :w] = img
        padded.append(canvas)

    rows = []
    for i in range(0, len(padded), cols):
        row_imgs = padded[i:i + cols]
        while len(row_imgs) < cols:
            row_imgs.append(np.full((max_h, max_w, 3), bg_color, dtype=np.uint8))
        rows.append(cv2.hconcat(row_imgs))

    return cv2.vconcat(rows)

def generate_synthetic_images(output_dir: str):
    ensure_dir(output_dir)

    img1 = np.zeros((256, 256), dtype=np.uint8)
    cv2.rectangle(img1, (40, 40), (120, 180), 255, -1)
    cv2.circle(img1, (185, 80), 35, 180, -1)
    cv2.line(img1, (20, 220), (230, 200), 220, 3)
    cv2.imwrite(os.path.join(output_dir, "synthetic_shapes.png"), img1)

    img2 = np.full((256, 256), 255, dtype=np.uint8)
    cv2.putText(img2, "EDGE", (25, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.5, 0, 3, cv2.LINE_AA)
    cv2.putText(img2, "NOISE", (18, 170), cv2.FONT_HERSHEY_SIMPLEX, 1.2, 50, 2, cv2.LINE_AA)
    cv2.imwrite(os.path.join(output_dir, "synthetic_text.png"), img2)

    x = np.linspace(0, 255, 256, dtype=np.float32)
    y = np.linspace(0, 255, 256, dtype=np.float32)
    xv, yv = np.meshgrid(x, y)
    img3 = 0.6 * xv + 0.4 * yv
    img3 = img3.astype(np.uint8)
    cv2.circle(img3, (128, 128), 50, 40, -1)
    cv2.rectangle(img3, (150, 30), (220, 100), 230, -1)
    cv2.imwrite(os.path.join(output_dir, "synthetic_pattern.png"), img3)


def load_or_create_images(input_dir: str):
    ensure_dir(input_dir)
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(exts)]

    if not image_files:
        print("[INFO] input_images 폴더에 이미지가 없어 합성 이미지를 생성합니다.")
        generate_synthetic_images(input_dir)
        image_files = [f for f in os.listdir(input_dir) if f.lower().endswith(exts)]

    images = []
    for fname in sorted(image_files):
        path = os.path.join(input_dir, fname)
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            print(f"[WARNING] 이미지를 읽을 수 없습니다: {path}")
            continue
        images.append((os.path.splitext(fname)[0], img))
    return images

def add_gaussian_noise(img: np.ndarray, sigma: float) -> np.ndarray:
    noise = np.random.normal(0, sigma, img.shape).astype(np.float32)
    noisy = img.astype(np.float32) + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper_noise(img: np.ndarray, amount: float) -> np.ndarray:
    noisy = img.copy()
    total = img.size
    num_salt = int(total * amount / 2)
    num_pepper = int(total * amount / 2)

    coords = (
        np.random.randint(0, img.shape[0], num_salt),
        np.random.randint(0, img.shape[1], num_salt)
    )
    noisy[coords] = 255

    coords = (
        np.random.randint(0, img.shape[0], num_pepper),
        np.random.randint(0, img.shape[1], num_pepper)
    )
    noisy[coords] = 0

    return noisy

# 전진 차분
def forward_diff(img: np.ndarray):
    img = img.astype(np.float32)
    gx = np.zeros_like(img, dtype=np.float32)
    gy = np.zeros_like(img, dtype=np.float32)
    gx[:, :-1] = img[:, 1:] - img[:, :-1]
    gy[:-1, :] = img[1:, :] - img[:-1, :]
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return gx, gy, mag

# 중심 차분
def central_diff(img: np.ndarray):
    img = img.astype(np.float32)
    gx = np.zeros_like(img, dtype=np.float32)
    gy = np.zeros_like(img, dtype=np.float32)
    gx[:, 1:-1] = (img[:, 2:] - img[:, :-2]) / 2.0
    gy[1:-1, :] = (img[2:, :] - img[:-2, :]) / 2.0
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return gx, gy, mag

# Prewitt 필터
def prewitt(img: np.ndarray):
    img = img.astype(np.float32)
    kx = np.array([[-1, 0, 1],
                   [-1, 0, 1],
                   [-1, 0, 1]], dtype=np.float32)
    ky = np.array([[-1, -1, -1],
                   [ 0,  0,  0],
                   [ 1,  1,  1]], dtype=np.float32)
    gx = cv2.filter2D(img, cv2.CV_32F, kx)
    gy = cv2.filter2D(img, cv2.CV_32F, ky)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return gx, gy, mag


# Sobel 필터
def sobel(img: np.ndarray):
    img = img.astype(np.float32)
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = np.sqrt(gx ** 2 + gy ** 2)
    return gx, gy, mag


METHODS = {
    "forward_diff": forward_diff,
    "central_diff": central_diff,
    "prewitt": prewitt,
    "sobel": sobel,
}

def run_experiment():
    ensure_dir(OUTPUT_DIR)
    images = load_or_create_images(INPUT_DIR)

    summary_rows = []

    clean_dir = os.path.join(OUTPUT_DIR, "clean")
    noisy_dir = os.path.join(OUTPUT_DIR, "noisy")
    sheet_dir = os.path.join(OUTPUT_DIR, "contact_sheets")
    ensure_dir(clean_dir)
    ensure_dir(noisy_dir)
    ensure_dir(sheet_dir)

    for image_name, img in images:
        print(f"[INFO] Processing image: {image_name}")

        clean_edges_by_method = {}
        clean_mags_by_method = {}

        base_panels = [put_title(img, f"{image_name} - original")]

        for method_name, method_func in METHODS.items():
            gx, gy, mag = method_func(img)
            mag_u8 = normalize_to_uint8(mag)
            edge = threshold_edge(mag)

            clean_mags_by_method[method_name] = mag_u8
            clean_edges_by_method[method_name] = edge

            method_dir = os.path.join(clean_dir, image_name, method_name)
            ensure_dir(method_dir)

            cv2.imwrite(os.path.join(method_dir, "magnitude.png"), mag_u8)
            cv2.imwrite(os.path.join(method_dir, "edge_binary.png"), edge)

            base_panels.append(put_title(mag_u8, f"{method_name} mag"))
            base_panels.append(put_title(edge, f"{method_name} edge"))

            summary_rows.append({
                "image": image_name,
                "condition": "clean",
                "noise_type": "none",
                "noise_level": 0,
                "method": method_name,
                "edge_pixels": int(np.sum(edge > 0)),
                "xor_diff_vs_clean": 0
            })

        clean_sheet = make_contact_sheet(base_panels, cols=3)
        cv2.imwrite(os.path.join(sheet_dir, f"{image_name}_clean_sheet.png"), clean_sheet)

        noisy_conditions = []

        for sigma in GAUSSIAN_SIGMAS:
            noisy_conditions.append(("gaussian", sigma, add_gaussian_noise(img, sigma)))

        for amount in SP_AMOUNTS:
            noisy_conditions.append(("salt_pepper", amount, add_salt_pepper_noise(img, amount)))

        for noise_type, noise_level, noisy_img in noisy_conditions:
            cond_name = f"{noise_type}_{str(noise_level).replace('.', '_')}"
            print(f"  [INFO] Noise condition: {cond_name}")

            panels = [
                put_title(img, f"{image_name} original"),
                put_title(noisy_img, f"{cond_name}")
            ]

            for method_name, method_func in METHODS.items():
                gx, gy, mag = method_func(noisy_img)
                mag_u8 = normalize_to_uint8(mag)
                edge = threshold_edge(mag)

                ref_edge = clean_edges_by_method[method_name]
                xor_diff = cv2.bitwise_xor(ref_edge, edge)
                xor_pixels = int(np.sum(xor_diff > 0))

                method_dir = os.path.join(noisy_dir, image_name, cond_name, method_name)
                ensure_dir(method_dir)

                cv2.imwrite(os.path.join(method_dir, "noisy_input.png"), noisy_img)
                cv2.imwrite(os.path.join(method_dir, "magnitude.png"), mag_u8)
                cv2.imwrite(os.path.join(method_dir, "edge_binary.png"), edge)
                cv2.imwrite(os.path.join(method_dir, "xor_vs_clean.png"), xor_diff)

                panels.append(put_title(mag_u8, f"{method_name} mag"))
                panels.append(put_title(edge, f"{method_name} edge"))
                panels.append(put_title(xor_diff, f"{method_name} xor"))

                summary_rows.append({
                    "image": image_name,
                    "condition": cond_name,
                    "noise_type": noise_type,
                    "noise_level": noise_level,
                    "method": method_name,
                    "edge_pixels": int(np.sum(edge > 0)),
                    "xor_diff_vs_clean": xor_pixels
                })

            sheet = make_contact_sheet(panels, cols=3)
            cv2.imwrite(os.path.join(sheet_dir, f"{image_name}_{cond_name}_sheet.png"), sheet)

    csv_path = os.path.join(OUTPUT_DIR, "summary_metrics.csv")
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image", "condition", "noise_type", "noise_level",
                "method", "edge_pixels", "xor_diff_vs_clean"
            ]
        )
        writer.writeheader()
        writer.writerows(summary_rows)

    print(f"[DONE] 결과 저장 완료: {OUTPUT_DIR}")
    print(f"[DONE] 요약 CSV: {csv_path}")


if __name__ == "__main__":
    run_experiment()
