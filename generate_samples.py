"""
Generates realistic synthetic test images of Indian Rupee (INR) coins.
Used for immediate testing, validation, and demonstrations in the web app.
Simulates accurate relative diameters, metallic textures, and surface shading.
"""

import os
import cv2
import numpy as np
from core.config import COIN_SPECS

def draw_realistic_coin(
    canvas: np.ndarray,
    cx: int,
    cy: int,
    denom: int,
    radius: int,
    shadow: bool = True
) -> None:
    """Draws a photorealistic coin with metallic luster, rim, and engraved text."""
    h, w = canvas.shape[:2]

    # 1. Soft Drop Shadow
    if shadow:
        shadow_mask = np.zeros((h, w), dtype=np.float32)
        cv2.circle(shadow_mask, (cx + 5, cy + 6), radius + 2, 0.45, -1)
        shadow_blur = cv2.GaussianBlur(shadow_mask, (15, 15), 5)
        for c in range(3):
            canvas[:, :, c] = (canvas[:, :, c] * (1.0 - shadow_blur)).astype(np.uint8)

    # Coordinate grid for radial and angular gradients
    Y, X = np.ogrid[:h, :w]
    dist = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    angle = np.arctan2(Y - cy, X - cx)

    coin_mask = dist <= radius
    if not np.any(coin_mask):
        return

    # Base metallic color profiles
    if denom == 1:       # Ferritic Stainless Steel (₹1: 20mm)
        base_color = np.array([210, 210, 215], dtype=np.float32)
        sheen_color = np.array([245, 245, 250], dtype=np.float32)
        is_bimetallic = False
    elif denom == 2:     # Stainless Steel (₹2: 25mm)
        base_color = np.array([205, 205, 210], dtype=np.float32)
        sheen_color = np.array([240, 240, 245], dtype=np.float32)
        is_bimetallic = False
    elif denom == 5:     # Nickel-Brass / Golden (₹5: 23mm)
        base_color = np.array([55, 180, 220], dtype=np.float32)   # Golden BGR
        sheen_color = np.array([90, 220, 255], dtype=np.float32)
        is_bimetallic = False
    elif denom == 10:    # Bimetallic (₹10: 27mm - Al-Bronze outer, Ni-Silver core)
        outer_color = np.array([50, 175, 215], dtype=np.float32)  # Golden outer ring
        outer_sheen = np.array([85, 215, 250], dtype=np.float32)
        inner_color = np.array([205, 205, 210], dtype=np.float32) # Silver core
        inner_sheen = np.array([240, 240, 245], dtype=np.float32)
        is_bimetallic = True
    else:                # ₹20
        outer_color = np.array([45, 170, 210], dtype=np.float32)
        outer_sheen = np.array([80, 210, 245], dtype=np.float32)
        inner_color = np.array([210, 210, 215], dtype=np.float32)
        inner_sheen = np.array([245, 245, 250], dtype=np.float32)
        is_bimetallic = True

    # Shading calculation (Light coming from top-left: -45 deg)
    light_dir = -np.pi / 4.0
    anisotropic_sheen = 0.5 * (1.0 + np.cos(2 * (angle - light_dir)))
    radial_dimming = 1.0 - 0.25 * (dist / radius) ** 2

    coin_patch = np.zeros((h, w, 3), dtype=np.float32)

    if not is_bimetallic:
        for c in range(3):
            val = (base_color[c] * (1.0 - 0.3 * anisotropic_sheen) + sheen_color[c] * (0.3 * anisotropic_sheen)) * radial_dimming
            coin_patch[:, :, c] = val
    else:
        # Bimetallic coin: Outer ring vs Inner core
        core_r = 0.58 * radius
        outer_mask = (dist <= radius) & (dist > core_r)
        inner_mask = dist <= core_r

        for c in range(3):
            # Outer ring
            outer_val = (outer_color[c] * (1.0 - 0.3 * anisotropic_sheen) + outer_sheen[c] * (0.3 * anisotropic_sheen)) * radial_dimming
            # Inner core
            inner_val = (inner_color[c] * (1.0 - 0.3 * anisotropic_sheen) + inner_sheen[c] * (0.3 * anisotropic_sheen)) * radial_dimming

            coin_patch[:, :, c] = outer_val * outer_mask + inner_val * inner_mask

    # Outer Raised Rim and Inner Engravings
    rim_outer = (dist <= radius) & (dist >= radius - 3)
    rim_inner = (dist <= radius - 3) & (dist >= radius - 5)

    coin_patch[rim_outer] = coin_patch[rim_outer] * 0.85
    coin_patch[rim_inner] = coin_patch[rim_inner] * 1.15

    if is_bimetallic:
        # Seam between outer ring and inner core
        seam = (dist <= 0.58 * radius + 1) & (dist >= 0.58 * radius - 1)
        coin_patch[seam] = coin_patch[seam] * 0.75

    # Clip values and assign to canvas
    coin_patch = np.clip(coin_patch, 0, 255).astype(np.uint8)
    canvas[coin_mask] = coin_patch[coin_mask]

    # Engraved Denomination Text in Center
    font = cv2.FONT_HERSHEY_DUPLEX
    font_scale = max(0.5, radius / 45.0)
    text = str(denom)
    (tw, th), _ = cv2.getTextSize(text, font, font_scale, 2)
    tx = cx - tw // 2
    ty = cy + th // 2

    # Engraved bevel effect (shadow + highlight)
    cv2.putText(canvas, text, (tx + 1, ty + 1), font, font_scale, (70, 70, 70), 2, cv2.LINE_AA)
    cv2.putText(canvas, text, (tx - 1, ty - 1), font, font_scale, (255, 255, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas, text, (tx, ty), font, font_scale, (120, 120, 130), 2, cv2.LINE_AA)

    # Subtext "RUPEES" / "RS"
    subfont = cv2.FONT_HERSHEY_SIMPLEX
    subscale = font_scale * 0.35
    subtext = "INDIA"
    (stw, sth), _ = cv2.getTextSize(subtext, subfont, subscale, 1)
    cv2.putText(canvas, subtext, (cx - stw // 2, cy - radius // 3), subfont, subscale, (100, 100, 110), 1, cv2.LINE_AA)

def create_wood_surface(width: int = 800, height: int = 600) -> np.ndarray:
    """Generates a soft, realistic desktop surface with warm neutral tones."""
    np.random.seed(42)
    # Neutral matte table background
    table = np.full((height, width, 3), [225, 228, 232], dtype=np.uint8)

    # Add gentle noise texture
    noise = np.random.normal(0, 3, (height, width, 3)).astype(np.int16)
    table = np.clip(table.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Gentle radial vignette
    Y, X = np.ogrid[:height, :width]
    vignette = 1.0 - 0.20 * (((X - width / 2) / (width / 2)) ** 2 + ((Y - height / 2) / (height / 2)) ** 2)
    vignette = np.clip(vignette, 0.7, 1.0)[:, :, np.newaxis]
    table = (table * vignette).astype(np.uint8)

    return table

def generate_sample_suite(output_dir: str = "static/samples"):
    """Generates three standard test images with accurate relative diameters."""
    os.makedirs(output_dir, exist_ok=True)

    # Scale: ~4 pixels per mm
    # ₹1:  20mm -> radius ~ 40px
    # ₹5:  23mm -> radius ~ 46px
    # ₹2:  25mm -> radius ~ 50px
    # ₹10: 27mm -> radius ~ 54px

    # --- Sample 1: Mixed Indian Coins (₹1, ₹2, ₹5, ₹10, ₹10) -> Total: ₹28 ---
    img1 = create_wood_surface(800, 600)
    coins_s1 = [
        (160, 180, 10, 54),  # ₹10 (27mm)
        (380, 170, 5, 46),   # ₹5  (23mm, Golden)
        (590, 190, 2, 50),   # ₹2  (25mm, Silver)
        (250, 410, 1, 40),   # ₹1  (20mm, Silver)
        (480, 420, 10, 54),  # ₹10 (27mm)
        (660, 400, 5, 46),   # ₹5  (23mm, Golden)
    ]
    for cx, cy, denom, r in coins_s1:
        draw_realistic_coin(img1, cx, cy, denom, r)

    p1 = os.path.join(output_dir, "sample_mix.jpg")
    cv2.imwrite(p1, img1)
    print(f"Generated: {p1}")

    # --- Sample 2: Scattered Assorted Coins ---
    img2 = create_wood_surface(800, 600)
    coins_s2 = [
        (150, 150, 1, 40),
        (320, 130, 2, 50),
        (510, 160, 5, 46),
        (680, 220, 10, 54),
        (200, 360, 5, 46),
        (390, 330, 10, 54),
        (580, 370, 2, 50),
        (300, 490, 1, 40),
        (500, 500, 1, 40),
    ]
    for cx, cy, denom, r in coins_s2:
        draw_realistic_coin(img2, cx, cy, denom, r)

    p2 = os.path.join(output_dir, "sample_scattered.jpg")
    cv2.imwrite(p2, img2)
    print(f"Generated: {p2}")

    # --- Sample 3: Touching / Clustered Coins (Watershed test) ---
    img3 = create_wood_surface(800, 600)
    # Touching coins: centers placed close so rims touch
    coins_s3 = [
        (330, 260, 10, 54),
        (434, 260, 5, 46),   # dist = 104, r1+r2 = 100 -> touching!
        (380, 350, 2, 50),   # touching both
        (580, 320, 1, 40),
        (210, 320, 5, 46),
    ]
    for cx, cy, denom, r in coins_s3:
        draw_realistic_coin(img3, cx, cy, denom, r)

    p3 = os.path.join(output_dir, "sample_touching.jpg")
    cv2.imwrite(p3, img3)
    print(f"Generated: {p3}")

if __name__ == "__main__":
    generate_sample_suite()
