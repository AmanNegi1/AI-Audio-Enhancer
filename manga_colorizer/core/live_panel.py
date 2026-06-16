import io
import math
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

from core.splitters import apply_splitter


LIVE_PANEL_TEMPLATES = [
    "None",
    "Cinematic Float",
    "Breathing Zoom",
    "Newspaper Wiggle",
    "Bass Shake Zoom",
    "Smooth Slow Drift",
    "3D Tilt Orbit",
    "3D Motion Fly Left",
    "3D Motion Fly Right",
    "Splitter Drift",
    "Panel Pop Pulse",
    "Glitch Pulse",
    "Smart Auto",
]

EFFECT_TEMPLATES = [
    "None",
    "Deep Glow",
    "Chromatic Aberration",
    "Manga Particles",
    "3D Depth Pop",
    "Stereo Edge 3D",
    "3D Corner Pin",
    "Screen Flip 3D",
    "Parallax Warp",
    "Parallax Drift",
    "Luxe Bloom",
    "Holographic Sheen",
    "Film Grain Glow",
    "Neon Edge",
    "Premium Glass",
    "Golden Hour Luxe",
    "Crystal Refraction",
    "Velvet Matte",
    "Prism Flare",
    "Smart Premium",
]


def _warp_full_image(
    image_rgb: np.ndarray,
    dx: float = 0.0,
    dy: float = 0.0,
    scale: float = 1.0,
    angle: float = 0.0,
    brightness: float = 1.0,
) -> np.ndarray:
    height, width = image_rgb.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, angle, scale)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    warped = cv2.warpAffine(
        image_rgb,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    ).astype(np.float32)
    if abs(brightness - 1.0) > 1e-6:
        warped *= brightness
    return np.clip(warped, 0, 255).astype(np.uint8)


def _base_motion(template: str, phase: float, strength: float) -> Dict[str, float]:
    sin1 = math.sin(phase)
    sin2 = math.sin(phase * 2.0)
    cos1 = math.cos(phase)

    if template == "3D Tilt Orbit":
        return {
            'dx': 4.5 * strength * sin1,
            'dy': 3.0 * strength * cos1,
            'scale': 1.0 + 0.020 * strength * (0.5 + 0.5 * cos1),
            'angle': 2.4 * strength * sin1,
            'brightness': 1.0 + 0.012 * strength * (0.5 + 0.5 * sin2),
        }

    if template == "3D Motion Fly Left":
        entry = 0.5 + 0.5 * sin1
        return {
            'dx': -7.5 * strength * entry,
            'dy': 1.4 * strength * math.sin(phase * 0.7),
            'scale': 1.0 + 0.040 * strength * entry,
            'angle': -1.4 * strength * (0.35 + 0.65 * entry),
            'brightness': 1.0 + 0.010 * strength * entry,
        }

    if template == "3D Motion Fly Right":
        entry = 0.5 + 0.5 * sin1
        return {
            'dx': 7.5 * strength * entry,
            'dy': 1.4 * strength * math.sin(phase * 0.7),
            'scale': 1.0 + 0.040 * strength * entry,
            'angle': 1.4 * strength * (0.35 + 0.65 * entry),
            'brightness': 1.0 + 0.010 * strength * entry,
        }

    if template == "Smooth Slow Drift":
        return {
            'dx': 1.6 * strength * sin1,
            'dy': -1.8 * strength * cos1,
            'scale': 1.0 + 0.010 * strength * (0.5 + 0.5 * sin1),
            'angle': 0.22 * strength * sin1,
            'brightness': 1.0 + 0.008 * strength * cos1,
        }

    if template == "Breathing Zoom":
        pulse = 0.5 + 0.5 * sin1
        return {
            'dx': 0.0,
            'dy': -3.0 * strength * pulse,
            'scale': 1.0 + 0.028 * strength * pulse,
            'angle': 0.45 * strength * sin1,
            'brightness': 1.0 + 0.025 * strength * pulse,
        }
    if template == "Newspaper Wiggle":
        return {
            'dx': 3.2 * strength * sin1,
            'dy': 2.0 * strength * cos1,
            'scale': 1.0 + 0.012 * strength * sin2,
            'angle': 1.3 * strength * sin2,
            'brightness': 1.0 + 0.012 * strength * sin1,
        }
    if template == "Bass Shake Zoom":
        hit = max(0.0, math.sin(phase * 2.0))
        return {
            'dx': (2.0 * sin1 + 5.0 * hit) * strength,
            'dy': (1.2 * cos1 - 4.0 * hit) * strength,
            'scale': 1.0 + (0.010 + 0.035 * hit) * strength,
            'angle': (0.4 * sin1 + 1.8 * hit) * strength,
            'brightness': 1.0 + (0.010 + 0.025 * hit) * strength,
        }
    return {
        'dx': 2.5 * strength * sin1,
        'dy': -2.5 * strength * (0.5 + 0.5 * sin1),
        'scale': 1.0 + 0.016 * strength * cos1,
        'angle': 0.5 * strength * sin1,
        'brightness': 1.0 + 0.010 * strength * sin1,
    }


def _smart_splitter_choice(splitter_template: str, panels: Optional[List[Dict]]) -> str:
    if splitter_template and splitter_template != "None":
        return splitter_template
    if panels:
        return "Panel Pop-out"
    return "Inset Grid"


def _soft_light_blend(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    base_f = base.astype(np.float32) / 255.0
    over_f = overlay.astype(np.float32) / 255.0
    soft = (1.0 - 2.0 * over_f) * (base_f ** 2) + 2.0 * over_f * base_f
    out = (1.0 - alpha) * base_f + alpha * soft
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _edge_mask(image_rgb: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 60, 160)
    edges = cv2.GaussianBlur(edges, (0, 0), 1.4)
    return edges.astype(np.float32) / 255.0


def _radial_mask(height: int, width: int, power: float = 1.6) -> np.ndarray:
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    cx = width / 2.0
    cy = height / 2.0
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    rr /= max(1.0, math.hypot(cx, cy))
    return np.clip(1.0 - rr, 0.0, 1.0) ** power


def _screen_blend(base: np.ndarray, overlay: np.ndarray, alpha: float) -> np.ndarray:
    base_f = base.astype(np.float32) / 255.0
    over_f = overlay.astype(np.float32) / 255.0
    screen = 1.0 - (1.0 - base_f) * (1.0 - over_f)
    out = (1.0 - alpha) * base_f + alpha * screen
    return np.clip(out * 255.0, 0, 255).astype(np.uint8)


def _temporal_blend(previous: np.ndarray, current: np.ndarray, alpha: float) -> np.ndarray:
    alpha = float(max(0.0, min(1.0, alpha)))
    return cv2.addWeighted(previous, 1.0 - alpha, current, alpha, 0)


def _make_particle_overlay(height: int, width: int, phase: float, strength: float) -> np.ndarray:
    overlay = np.zeros((height, width, 3), dtype=np.uint8)
    count = 20 + int(45 * strength)
    seed = int(round((phase % (2.0 * math.pi)) * 10000)) + 101
    rng = np.random.default_rng(seed)
    for _ in range(count):
        x = int(rng.integers(0, width))
        y = int(rng.integers(0, height))
        radius = int(rng.integers(1, max(2, int(5 + 6 * strength))))
        hue_pick = float(rng.random())
        if hue_pick < 0.33:
            color = (255, 245, 220)
        elif hue_pick < 0.66:
            color = (255, 210, 150)
        else:
            color = (180, 220, 255)
        cv2.circle(overlay, (x, y), radius, color, -1)
    blur_sigma = 1.5 + 3.5 * strength
    return cv2.GaussianBlur(overlay, (0, 0), blur_sigma)


def _perspective_warp(
    frame: np.ndarray,
    phase: float,
    strength: float,
    x_gain: float = 1.0,
    y_gain: float = 1.0,
    corner_gain: float = 0.6,
    y_phase_scale: float = 1.0,
    y_phase_offset: float = 0.0,
) -> np.ndarray:
    height, width = frame.shape[:2]
    dx = width * (0.018 + 0.030 * strength) * x_gain * math.sin(phase)
    dy = height * (0.012 + 0.022 * strength) * y_gain * math.sin((phase * y_phase_scale) + y_phase_offset)
    src = np.float32([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1],
    ])
    dst = np.float32([
        [0 + dx, 0 + dy],
        [width - 1 - dx, 0 - dy],
        [width - 1 + dx * corner_gain, height - 1 - dy],
        [0 - dx * corner_gain, height - 1 + dy],
    ])
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        frame,
        matrix,
        (width, height),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT_101,
    )


def _apply_effect(
    frame: np.ndarray,
    effect_template: str,
    phase: float,
    strength: float,
) -> np.ndarray:
    if effect_template == "None":
        return frame

    height, width = frame.shape[:2]
    sin1 = math.sin(phase)
    cos1 = math.cos(phase)
    strength = float(max(0.0, min(1.0, strength)))

    if effect_template == "Smart Premium":
        effect_template = "Premium Glass"

    if effect_template == "3D Depth Pop":
        warped = _perspective_warp(frame, phase, strength)
        sharp = cv2.addWeighted(warped, 1.20, cv2.GaussianBlur(warped, (0, 0), 2.2 + 2.8 * strength), -0.20, 0)
        vignette = _radial_mask(height, width, power=1.1)
        out = sharp.astype(np.float32)
        out *= (0.86 + 0.18 * vignette[..., None])
        return np.clip(out, 0, 255).astype(np.uint8)

    if effect_template == "3D Corner Pin":
        pinned = _perspective_warp(
            frame,
            phase,
            strength,
            x_gain=0.65,
            y_gain=0.38,
            corner_gain=1.10,
            y_phase_scale=0.65,
            y_phase_offset=0.55,
        )
        glossy = cv2.GaussianBlur(pinned, (0, 0), 1.2 + 1.8 * strength)
        mixed = cv2.addWeighted(pinned, 0.86, glossy, 0.14, 0)
        vignette = _radial_mask(height, width, power=1.3)[..., None]
        out = mixed.astype(np.float32)
        out *= (0.84 + 0.22 * vignette)
        return np.clip(out, 0, 255).astype(np.uint8)

    if effect_template == "Screen Flip 3D":
        turn = 0.5 - 0.5 * math.cos(phase)
        angle = turn * math.pi
        side = 1.0 if angle <= (math.pi * 0.5) else -1.0
        edge_amount = math.sin(angle)
        width_scale = max(0.20, abs(math.cos(angle)))
        scaled_w = max(2, int(round(width * width_scale)))
        resized = cv2.resize(frame, (scaled_w, height), interpolation=cv2.INTER_LINEAR)
        if side < 0.0:
            resized = cv2.flip(resized, 1)

        canvas = np.zeros_like(frame)
        x1 = (width - scaled_w) // 2
        x2 = x1 + scaled_w
        canvas[:, x1:x2] = resized

        src = np.float32([
            [x1, 0],
            [x2 - 1, 0],
            [x2 - 1, height - 1],
            [x1, height - 1],
        ])
        skew = width * (0.04 + 0.10 * strength) * edge_amount
        y_inset = height * (0.02 + 0.06 * strength) * edge_amount
        if side > 0.0:
            dst = np.float32([
                [x1 + skew, y_inset],
                [x2 - 1, 0],
                [x2 - 1 - skew, height - 1 - y_inset],
                [x1, height - 1],
            ])
        else:
            dst = np.float32([
                [x1, 0],
                [x2 - 1 - skew, y_inset],
                [x2 - 1, height - 1],
                [x1 + skew, height - 1 - y_inset],
            ])
        matrix = cv2.getPerspectiveTransform(src, dst)
        turned = cv2.warpPerspective(
            canvas,
            matrix,
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )

        shade = 0.76 + 0.24 * width_scale
        out = turned.astype(np.float32) * shade

        edge_glow = np.zeros((height, width), dtype=np.float32)
        edge_band = max(2, int(width * 0.030))
        edge_center = width // 2
        glow_left = max(0, edge_center - edge_band)
        glow_right = min(width, edge_center + edge_band)
        if glow_right > glow_left:
            ramp = np.linspace(0.0, 1.0, glow_right - glow_left, dtype=np.float32)
            ramp = np.minimum(ramp, ramp[::-1]) * 2.0
            edge_glow[:, glow_left:glow_right] = ramp[None, :] * edge_amount

        glow_color = np.array([55.0, 95.0, 150.0], dtype=np.float32)
        out += edge_glow[..., None] * glow_color * (0.12 + 0.28 * strength)

        vignette = _radial_mask(height, width, power=1.5)[..., None]
        out *= (0.84 + 0.16 * vignette)
        return np.clip(out, 0, 255).astype(np.uint8)

    if effect_template == "Stereo Edge 3D":
        shift = max(1, int((2.0 + 7.0 * strength) * (0.5 + 0.5 * math.sin(phase))))
        red = np.roll(frame[..., 0], shift, axis=1)
        cyan_g = np.roll(frame[..., 1], -shift, axis=1)
        cyan_b = np.roll(frame[..., 2], -shift, axis=1)
        mixed = np.dstack([red, cyan_g, cyan_b]).astype(np.uint8)
        edge = _edge_mask(frame)
        edge_mix = frame.astype(np.float32) * (1.0 - 0.22 * edge[..., None]) + mixed.astype(np.float32) * (0.22 * edge[..., None] + 0.18 * strength)
        return np.clip(edge_mix, 0, 255).astype(np.uint8)

    if effect_template == "Parallax Warp":
        warped = _perspective_warp(
            frame,
            phase,
            strength,
            x_gain=0.42,
            y_gain=0.24,
            corner_gain=0.32,
            y_phase_scale=0.55,
            y_phase_offset=0.8,
        )
        depth_mask = _radial_mask(height, width, power=1.7)[..., None]
        zoom = _warp_full_image(
            frame,
            scale=1.0 + 0.010 * strength,
            dx=0.55 * strength * math.sin(phase * 0.7),
            dy=-0.35 * strength * math.cos(phase * 0.55),
        )
        out = warped.astype(np.float32) * (1.0 - 0.14 * depth_mask) + zoom.astype(np.float32) * (0.14 * depth_mask)
        return np.clip(out, 0, 255).astype(np.uint8)

    if effect_template == "Parallax Drift":
        warped = _perspective_warp(
            frame,
            phase,
            strength,
            x_gain=0.24,
            y_gain=0.12,
            corner_gain=0.20,
            y_phase_scale=0.42,
            y_phase_offset=1.2,
        )
        depth_mask = _radial_mask(height, width, power=2.0)[..., None]
        zoom = _warp_full_image(
            frame,
            scale=1.0 + 0.006 * strength,
            dx=0.22 * strength * math.sin(phase * 0.55),
            dy=-0.16 * strength * math.cos(phase * 0.45),
        )
        out = warped.astype(np.float32) * (1.0 - 0.10 * depth_mask) + zoom.astype(np.float32) * (0.10 * depth_mask)
        return np.clip(out, 0, 255).astype(np.uint8)

    if effect_template == "Deep Glow":
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        thresh = int(170 - 45 * strength)
        mask = (gray > thresh).astype(np.uint8) * 255
        mask = cv2.GaussianBlur(mask, (0, 0), 4.0 + 8.0 * strength)
        glow = cv2.GaussianBlur(frame, (0, 0), 6.0 + 14.0 * strength)
        tint = np.zeros_like(glow, dtype=np.float32)
        tint[..., 0] = 255.0
        tint[..., 1] = 185.0 + 35.0 * (0.5 + 0.5 * sin1)
        tint[..., 2] = 235.0 + 20.0 * (0.5 + 0.5 * cos1)
        glow = np.clip(glow.astype(np.float32) * 0.55 + tint * 0.45, 0, 255).astype(np.uint8)
        alpha = (mask.astype(np.float32) / 255.0) * (0.25 + 0.45 * strength)
        out = frame.astype(np.float32) * (1.0 - alpha[..., None]) + glow.astype(np.float32) * alpha[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)

    if effect_template == "Chromatic Aberration":
        shift = max(1, int((1.0 + 7.0 * strength) * (0.6 + 0.4 * (0.5 + 0.5 * sin1))))
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        cx = width / 2.0
        cy = height / 2.0
        rx = (xx - cx) / max(1.0, cx)
        ry = (yy - cy) / max(1.0, cy)
        radial = np.clip(np.sqrt(rx ** 2 + ry ** 2), 0.0, 1.0)
        shift_map = radial * shift
        map_x_r = np.clip(xx + shift_map, 0, width - 1).astype(np.float32)
        map_x_b = np.clip(xx - shift_map, 0, width - 1).astype(np.float32)
        map_y = yy.astype(np.float32)
        red = cv2.remap(frame[..., 0], map_x_r, map_y, cv2.INTER_LINEAR)
        green = frame[..., 1]
        blue = cv2.remap(frame[..., 2], map_x_b, map_y, cv2.INTER_LINEAR)
        return np.dstack([red, green, blue]).astype(np.uint8)

    if effect_template == "Manga Particles":
        particles = _make_particle_overlay(height, width, phase, strength)
        return _screen_blend(frame, particles, 0.22 + 0.28 * strength)

    if effect_template == "Luxe Bloom":
        glow = cv2.GaussianBlur(frame, (0, 0), 4.0 + 8.0 * strength)
        alpha = 0.18 + 0.22 * strength * (0.5 + 0.5 * sin1)
        blended = cv2.addWeighted(frame, 1.0, glow, alpha, 0)
        return np.clip(blended, 0, 255).astype(np.uint8)

    if effect_template == "Holographic Sheen":
        xx = np.linspace(0.0, 1.0, width, dtype=np.float32)
        yy = np.linspace(0.0, 1.0, height, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xx, yy)
        band = np.sin((grid_x * 7.5) + phase) * 0.5 + 0.5
        cyan = np.dstack([
            0.30 + 0.20 * band,
            0.70 + 0.20 * band,
            0.85 + 0.10 * band,
        ])
        magenta = np.dstack([
            0.85 + 0.10 * (1.0 - band),
            0.35 + 0.15 * (1.0 - band),
            0.75 + 0.15 * (1.0 - band),
        ])
        mix = np.clip((0.55 * cyan + 0.45 * magenta) * (0.65 + 0.35 * grid_y[..., None]), 0.0, 1.0)
        overlay = (mix * 255).astype(np.uint8)
        return _soft_light_blend(frame, overlay, 0.18 + 0.18 * strength)

    if effect_template == "Film Grain Glow":
        rng = np.random.default_rng(int(round((phase % (2.0 * math.pi)) * 1000)) + 17)
        noise = rng.normal(0, 12 + 18 * strength, frame.shape).astype(np.float32)
        grainy = np.clip(frame.astype(np.float32) + noise, 0, 255).astype(np.uint8)
        glow = cv2.GaussianBlur(grainy, (0, 0), 2.0 + 3.0 * strength)
        return cv2.addWeighted(grainy, 0.88, glow, 0.12 + 0.12 * strength, 0)

    if effect_template == "Golden Hour Luxe":
        radial = _radial_mask(height, width, power=1.4)[..., None]
        warm = np.zeros_like(frame, dtype=np.float32)
        warm[..., 0] = 255.0
        warm[..., 1] = 198.0 + 18.0 * math.sin(phase)
        warm[..., 2] = 126.0
        warmed = frame.astype(np.float32) * (1.0 - 0.10 * strength) + warm * radial * (0.16 + 0.20 * strength)
        bloom = cv2.GaussianBlur(np.clip(warmed, 0, 255).astype(np.uint8), (0, 0), 5.0 + 7.0 * strength)
        return cv2.addWeighted(np.clip(warmed, 0, 255).astype(np.uint8), 0.82, bloom, 0.18, 0)

    if effect_template == "Crystal Refraction":
        xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(xx, yy)
        offset = (np.sin((grid_y * 8.0) + phase) * (1.5 + 3.0 * strength)).astype(np.float32)
        map_x = np.clip(np.tile(np.arange(width, dtype=np.float32), (height, 1)) + offset, 0, width - 1)
        map_y = np.tile(np.arange(height, dtype=np.float32)[:, None], (1, width))
        refracted = cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
        sheen = cv2.GaussianBlur(refracted, (0, 0), 2.0 + 2.5 * strength)
        mixed = cv2.addWeighted(frame, 0.72, sheen, 0.28, 0)
        highlight = (_radial_mask(height, width, power=2.2) * (30.0 + 40.0 * strength)).astype(np.float32)
        out = mixed.astype(np.float32)
        out += highlight[..., None]
        return np.clip(out, 0, 255).astype(np.uint8)

    if effect_template == "Neon Edge":
        edges = _edge_mask(frame)[..., None]
        tint = np.zeros_like(frame, dtype=np.float32)
        tint[..., 0] = 255.0 * (0.65 + 0.25 * sin1)
        tint[..., 1] = 80.0 + 90.0 * (0.5 + 0.5 * cos1)
        tint[..., 2] = 255.0 * (0.80 + 0.15 * cos1)
        out = frame.astype(np.float32) + tint * edges * (0.35 + 0.45 * strength)
        return np.clip(out, 0, 255).astype(np.uint8)

    if effect_template == "Premium Glass":
        overlay = frame.astype(np.float32).copy()
        panel = cv2.GaussianBlur(frame, (0, 0), 8.0 + 10.0 * strength).astype(np.float32)
        overlay = cv2.addWeighted(overlay, 0.84, panel, 0.16, 0)
        band_x = int((0.18 + 0.58 * (0.5 + 0.5 * sin1)) * width)
        band_w = max(20, int(width * (0.08 + 0.10 * strength)))
        sheen = np.zeros((height, width), dtype=np.float32)
        x1 = max(0, band_x - band_w)
        x2 = min(width, band_x + band_w)
        if x2 > x1:
            ramp = np.linspace(0.0, 1.0, x2 - x1, dtype=np.float32)
            ramp = np.minimum(ramp, ramp[::-1]) * 2.0
            sheen[:, x1:x2] = ramp[None, :]
        overlay += sheen[..., None] * np.array([65.0, 70.0, 80.0], dtype=np.float32) * (0.8 + 0.4 * strength)
        vignette_x = np.linspace(-1.0, 1.0, width, dtype=np.float32)
        vignette_y = np.linspace(-1.0, 1.0, height, dtype=np.float32)
        grid_x, grid_y = np.meshgrid(vignette_x, vignette_y)
        vignette = np.clip(1.0 - 0.18 * strength * (grid_x ** 2 + grid_y ** 2), 0.72, 1.0)
        overlay *= vignette[..., None]
        return np.clip(overlay, 0, 255).astype(np.uint8)

    if effect_template == "Velvet Matte":
        blurred = cv2.GaussianBlur(frame, (0, 0), 3.5 + 4.0 * strength)
        matte = cv2.addWeighted(frame, 0.68, blurred, 0.32, 0)
        lab = cv2.cvtColor(matte, cv2.COLOR_RGB2LAB).astype(np.float32)
        lab[..., 1] = 128.0 + (lab[..., 1] - 128.0) * (0.92 - 0.20 * strength)
        lab[..., 2] = 128.0 + (lab[..., 2] - 128.0) * (0.96 - 0.16 * strength)
        matte = cv2.cvtColor(np.clip(lab, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)
        vignette = _radial_mask(height, width, power=1.2)
        matte_f = matte.astype(np.float32)
        matte_f *= (0.82 + 0.18 * vignette[..., None])
        return np.clip(matte_f, 0, 255).astype(np.uint8)

    if effect_template == "Prism Flare":
        out = frame.astype(np.float32)
        center_x = int((0.15 + 0.70 * (0.5 + 0.5 * math.sin(phase))) * width)
        flare_w = max(20, int(width * (0.05 + 0.10 * strength)))
        flare = np.zeros((height, width), dtype=np.float32)
        x1 = max(0, center_x - flare_w)
        x2 = min(width, center_x + flare_w)
        if x2 > x1:
            ramp = np.linspace(0.0, 1.0, x2 - x1, dtype=np.float32)
            ramp = np.minimum(ramp, ramp[::-1]) * 2.0
            flare[:, x1:x2] = ramp[None, :]
        colors = np.array([255.0, 180.0 + 35.0 * math.sin(phase), 120.0 + 70.0 * math.cos(phase)], dtype=np.float32)
        out += flare[..., None] * colors * (0.22 + 0.34 * strength)
        sparkle = cv2.GaussianBlur(flare, (0, 0), 8.0 + 8.0 * strength)
        out += sparkle[..., None] * np.array([80.0, 110.0, 160.0], dtype=np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

    return frame


def render_live_panel_frames(
    image_rgb: np.ndarray,
    template: str = "Cinematic Float",
    strength: float = 0.5,
    frame_count: int = 18,
    panels: Optional[List[Dict]] = None,
    splitter_template: str = "None",
    gap_color=(255, 255, 255),
    effect_template: str = "None",
    effect_strength: float = 0.45,
) -> List[np.ndarray]:
    """Generate whole-image animation loops from the final stylized page.

    The goal is a smarter, lower-friction animation system than the old
    subject-box approach: choose a template and animate the whole page using
    subtle camera motion plus optional splitter-based transforms.
    """
    if template == "None" and effect_template == "None":
        return [image_rgb.copy()]

    strength = float(max(0.0, min(1.0, strength)))
    frame_count = max(2, int(frame_count))
    gap_color = tuple(int(c) for c in gap_color)
    frames: List[np.ndarray] = []
    previous_frame: Optional[np.ndarray] = None

    chosen_splitter = _smart_splitter_choice(splitter_template, panels)

    for idx in range(frame_count):
        phase = (idx / frame_count) * 2.0 * math.pi
        motion_template = template if template in (
            "Cinematic Float", "Breathing Zoom", "Newspaper Wiggle",
            "Bass Shake Zoom", "Smooth Slow Drift", "3D Tilt Orbit",
            "3D Motion Fly Left", "3D Motion Fly Right"
        ) else "Cinematic Float"
        if template == "None":
            frame = image_rgb.copy()
        else:
            motion = _base_motion(motion_template, phase, strength)
            frame = _warp_full_image(
                image_rgb,
                dx=motion['dx'],
                dy=motion['dy'],
                scale=motion['scale'],
                angle=motion['angle'],
                brightness=motion['brightness'],
            )

        if template == "Splitter Drift":
            split_strength = 0.18 + 0.42 * strength * (0.5 + 0.5 * math.sin(phase))
            frame = apply_splitter(
                frame,
                template=chosen_splitter,
                intensity=split_strength,
                panels=panels,
                gap_color=gap_color,
            )
        elif template == "Panel Pop Pulse":
            split_strength = 0.10 + 0.55 * strength * (0.5 + 0.5 * math.sin(phase))
            frame = apply_splitter(
                frame,
                template="Panel Pop-out",
                intensity=split_strength,
                panels=panels,
                gap_color=gap_color,
            )
        elif template == "Glitch Pulse":
            pulse = 0.10 + 0.65 * strength * (0.5 + 0.5 * math.sin(phase * 2.0))
            if idx % 2 == 0:
                frame = apply_splitter(
                    frame,
                    template="Glitch Slice",
                    intensity=pulse,
                    panels=panels,
                    gap_color=gap_color,
                )
        elif template == "Smart Auto":
            split_strength = 0.15 + 0.45 * strength * (0.5 + 0.5 * math.sin(phase))
            frame = apply_splitter(
                frame,
                template=chosen_splitter,
                intensity=split_strength,
                panels=panels,
                gap_color=gap_color,
            )

        frame = _apply_effect(frame, effect_template, phase, effect_strength)

        if effect_template == "Parallax Warp" and previous_frame is not None:
            frame = _temporal_blend(previous_frame, frame, 0.32)
        elif effect_template == "Parallax Drift" and previous_frame is not None:
            frame = _temporal_blend(previous_frame, frame, 0.26)

        frames.append(frame)
        previous_frame = frame

    return frames


def encode_gif_bytes(frames: List[np.ndarray], fps: int = 12) -> bytes:
    fps = max(1, int(fps))
    duration_ms = int(round(1000 / fps))
    pil_frames = [Image.fromarray(frame) for frame in frames]
    buf = io.BytesIO()
    pil_frames[0].save(
        buf,
        format="GIF",
        save_all=True,
        append_images=pil_frames[1:],
        duration=duration_ms,
        loop=0,
        disposal=2,
    )
    return buf.getvalue()