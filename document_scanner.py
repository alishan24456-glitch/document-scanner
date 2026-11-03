"""
Document Scanner — edge detection + perspective-transform scanning.

One pipeline serves both modes:

* Real mode (``--source <video | 0 | pick>``): every incoming frame is
  searched for a document with edge detection + contour approximation,
  the four corners are warped with a perspective transform, and the
  result is enhanced into a clean scanned page.
* Simulation mode (``--simulate``): a skewed "photographed" document is
  rendered on a desk, then read back out of the rendered pixels by the
  exact same detector. The simulation never hands the detector the
  true corner positions — detection comes only from image pixels, just
  as real-mode detection comes only from camera pixels.

Running with no arguments (or ``--source pick``) opens a file-picker
dialog so the script runs directly from Windows CMD with a video
file-chooser, matching the standing delivery rule for this project set.

Usage (CMD / terminal)::

    pip install -r requirements.txt
    python document_scanner.py --simulate
    python document_scanner.py --simulate --output demo.mp4 --duration 28
    python document_scanner.py --source pick
    python document_scanner.py
    python document_scanner.py --source 0
    python document_scanner.py --source photo_of_receipt.mp4 --output scanned.mp4

Press ``q`` (or Esc) to stop a live window. ``--no-display`` skips the
preview window and only writes ``--output`` when given.
"""

import argparse
import sys
import time

import cv2
import numpy as np

FRAME_W, FRAME_H = 1280, 720
SCAN_W, SCAN_H = 500, 700          # warped "A4" page size
STABLE_FRAMES_FOR_SCAN = 45        # consecutive detections before a scan
FLASH_FRAMES = 45                  # how long the SCANNED banner shows


# ---------------------------------------------------------------- pipeline
def order_points(pts):
    """Order 4 corner points as top-left, top-right, bottom-right, bottom-left."""
    pts = np.array(pts, dtype="float32").reshape(4, 2)
    ordered = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    ordered[0] = pts[np.argmin(s)]          # top-left: smallest x + y
    ordered[2] = pts[np.argmax(s)]          # bottom-right: largest x + y
    diff = np.diff(pts, axis=1).ravel()
    ordered[1] = pts[np.argmin(diff)]       # top-right: smallest y - x
    ordered[3] = pts[np.argmax(diff)]       # bottom-left: largest y - x
    return ordered


def detect_document(frame):
    """Find the largest 4-point contour in a frame.

    Returns the ordered corner points, or None when no plausible
    document quadrilateral is visible. Judges pixels only — no input
    other than the frame itself.
    """
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    edges = cv2.dilate(edges, np.ones((3, 3), np.uint8), iterations=1)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)
    frame_area = frame.shape[0] * frame.shape[1]
    for contour in sorted(contours, key=cv2.contourArea, reverse=True)[:5]:
        area = cv2.contourArea(contour)
        if area < 0.12 * frame_area or area > 0.95 * frame_area:
            continue
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02 * peri, True)
        if len(approx) == 4:
            return order_points(approx)
    return None


def enhance_scan(warped):
    """Turn a warped colour page into a clean black-on-white scan."""
    gray = cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    gray = cv2.adaptiveThreshold(
        gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY, 21, 12)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def scan_frame(frame):
    """Run the shared pipeline on one frame.

    Returns ``(corners_or_None, scanned_page_or_None)``.
    """
    corners = detect_document(frame)
    if corners is None:
        return None, None
    dst = np.array([[0, 0], [SCAN_W - 1, 0],
                    [SCAN_W - 1, SCAN_H - 1], [0, SCAN_H - 1]],
                   dtype="float32")
    matrix = cv2.getPerspectiveTransform(corners, dst)
    warped = cv2.warpPerspective(frame, matrix, (SCAN_W, SCAN_H))
    return corners, enhance_scan(warped)


# ------------------------------------------------------------- simulation
def make_flat_page():
    """Draw the flat (unwarped) document content: text, table, chart."""
    page = np.full((850, 600, 3), 245, dtype=np.uint8)
    cv2.rectangle(page, (30, 30), (570, 95), (30, 30, 30), -1)
    cv2.putText(page, "QUARTERLY REPORT", (50, 78),
                cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    y = 140
    rng = np.random.default_rng(7)
    for _ in range(9):  # paragraph text lines
        length = int(rng.integers(320, 520))
        cv2.rectangle(page, (50, y), (50 + length, y + 10), (60, 60, 60), -1)
        y += 26
    y += 10
    for r in range(4):  # small table
        for c in range(3):
            x0 = 50 + c * 170
            cv2.rectangle(page, (x0, y), (x0 + 150, y + 28), (90, 90, 90), 1)
            cv2.rectangle(page, (x0 + 8, y + 9), (x0 + 110, y + 19),
                          (60, 60, 60), -1)
        y += 34
    y += 15
    cv2.putText(page, "GROWTH", (50, y), cv2.FONT_HERSHEY_SIMPLEX,
                0.6, (60, 60, 60), 2)
    base = y + 130
    for i, height in enumerate([45, 70, 55, 95, 120]):
        x0 = 60 + i * 95
        cv2.rectangle(page, (x0, base - height), (x0 + 55, base),
                      (40, 90, 160), -1)
    cv2.line(page, (50, base), (550, base), (60, 60, 60), 2)
    cv2.putText(page, "Page 1 of 1", (430, 815),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 120, 120), 1)
    return page


def render_scene(frame_idx, flat_page):
    """Render a desk + skewed photographed document for one frame.

    The page drifts and tilts gently over time. Only pixels leave this
    function — the detector never sees the corner positions.
    """
    rng = np.random.default_rng(42)
    desk = np.zeros((FRAME_H, FRAME_W, 3), dtype=np.uint8)
    desk[:] = (52, 72, 96)  # dark wood-brown desk (BGR)
    noise = rng.integers(-8, 9, (FRAME_H, FRAME_W, 1)).astype(np.int16)
    desk = np.clip(desk.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    for x in range(0, FRAME_W, 160):  # wood grain
        cv2.line(desk, (x, 0), (x + 40, FRAME_H), (44, 62, 82), 1)

    t = frame_idx / 30.0
    sway_x = 22 * np.sin(t * 0.5)
    sway_y = 12 * np.sin(t * 0.8 + 1.0)
    tilt = 10 * np.sin(t * 0.35)
    src = np.array([[0, 0], [599, 0], [599, 849], [0, 849]], dtype="float32")
    dst = np.array([
        [330 + sway_x - tilt, 70 + sway_y],
        [940 + sway_x + tilt, 95 + sway_y],
        [905 + sway_x, 650 + sway_y],
        [365 + sway_x, 625 + sway_y],
    ], dtype="float32")
    # soft drop shadow, offset down-right from the page
    shadow = desk.copy()
    cv2.fillPoly(shadow, [np.int32(dst + np.array([10, 12]))], (0, 0, 0))
    desk = cv2.addWeighted(shadow, 0.35, desk, 0.65, 0)
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(flat_page, matrix, (FRAME_W, FRAME_H))
    mask = cv2.warpPerspective(np.full((850, 600), 255, dtype=np.uint8),
                               matrix, (FRAME_W, FRAME_H))
    desk[mask > 0] = warped[mask > 0]
    return desk


# ---------------------------------------------------------------- overlay
def annotate(frame, corners, scanned, stats, flash_left):
    """Draw quad, scanned inset, HUD and SCANNED banner onto a frame."""
    out = frame.copy()
    if corners is not None:
        pts = corners.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(out, [pts], True, (0, 200, 0), 3)
        for (x, y) in corners.astype(int):
            cv2.circle(out, (x, y), 8, (0, 0, 255), -1)
        label = "DOCUMENT DETECTED"
        colour = (0, 200, 0)
    else:
        label = "NO DOCUMENT"
        colour = (0, 0, 255)
    cv2.putText(out, label, (30, 45), cv2.FONT_HERSHEY_SIMPLEX,
                1.0, colour, 2)
    hud = (f"Frames: {stats['frames']}   Detected: {stats['detected']} "
           f"({stats['rate']:.0f}%)   Scans: {stats['scans']}")
    cv2.putText(out, hud, (30, FRAME_H - 20), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (255, 255, 255), 2)
    if scanned is not None:
        inset = cv2.resize(scanned, (252, 352))
        x0, y0 = FRAME_W - 282, 70
        cv2.rectangle(out, (x0 - 4, y0 - 4), (x0 + 256, y0 + 356),
                      (255, 255, 255), 3)
        out[y0:y0 + 352, x0:x0 + 252] = inset
        cv2.putText(out, "SCANNED", (x0 + 55, y0 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    if flash_left > 0:
        cv2.rectangle(out, (340, 280), (940, 360), (0, 180, 0), -1)
        cv2.putText(out, "SCANNED ✓", (470, 332),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
    return out


# ------------------------------------------------------------------- main
def pick_video_file():
    """Tkinter file-picker (also auto-opens when run with no arguments)."""
    import tkinter as tk
    from tkinter import filedialog
    root = tk.Tk()
    root.withdraw()
    path = filedialog.askopenfilename(
        title="Choose a video file",
        filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv"),
                   ("All files", "*.*")])
    root.destroy()
    return path or None


def run(source, simulate, output, duration, no_display):
    if simulate:
        flat_page = make_flat_page()
        total = int(duration * 30)
        get_frame = lambda i: render_scene(i, flat_page)  # noqa: E731
        n_frames = total
    else:
        cap = cv2.VideoCapture(source)
        if not cap.isOpened():
            print(f"ERROR: could not open source {source!r}", file=sys.stderr)
            return None
        get_frame = lambda i: cap.read()[1]  # noqa: E731
        n_frames = None

    writer = None
    if output:
        writer = cv2.VideoWriter(output, cv2.VideoWriter_fourcc(*"mp4v"),
                                 30, (FRAME_W, FRAME_H))
        if not writer.isOpened():
            print(f"ERROR: could not open output {output!r}", file=sys.stderr)
            return None

    stats = {"frames": 0, "detected": 0, "scans": 0, "rate": 0.0}
    stable = 0
    longest_streak = 0
    scan_cooldown = 0  # frames until the next auto-capture is allowed
    flash_left = 0
    last_scanned = None
    idx = 0
    while True:
        if simulate and idx >= n_frames:
            break
        frame = get_frame(idx)
        if frame is None:
            break
        frame = cv2.resize(frame, (FRAME_W, FRAME_H))
        corners, scanned = scan_frame(frame)
        stats["frames"] += 1
        if corners is not None:
            stats["detected"] += 1
            stable += 1
            longest_streak = max(longest_streak, stable)
            last_scanned = scanned
            if scan_cooldown > 0:
                scan_cooldown -= 1
            if stable >= STABLE_FRAMES_FOR_SCAN and scan_cooldown == 0:
                stats["scans"] += 1
                scan_cooldown = 240  # next auto-capture after ~8 s
                flash_left = FLASH_FRAMES
        else:
            stable = 0
            scan_cooldown = 0
        if flash_left > 0:
            flash_left -= 1
        stats["rate"] = (100.0 * stats["detected"] / stats["frames"]
                         if stats["frames"] else 0.0)
        shown = annotate(frame, corners, last_scanned, stats, flash_left)
        if writer:
            writer.write(shown)
        if not no_display:
            cv2.imshow("Document Scanner  (q to quit)", shown)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
        idx += 1

    if writer:
        writer.release()
    if not simulate:
        cap.release()
    try:
        cv2.destroyAllWindows()  # no-op on headless opencv builds
    except cv2.error:
        pass
    print(f"Frames processed: {stats['frames']}")
    print(f"Frames with document detected: {stats['detected']} "
          f"({stats['rate']:.1f}%)")
    print(f"Scans captured (stable >= {STABLE_FRAMES_FOR_SCAN} frames): "
          f"{stats['scans']}")
    print(f"Longest stable detection streak: {longest_streak} frames")
    print(f"Final state: {'DOCUMENT DETECTED' if stable else 'NO DOCUMENT'}")
    return stats


def main(argv=None):
    parser = argparse.ArgumentParser(description="Document scanner")
    parser.add_argument("--source", default=None,
                        help="video path, webcam index, or 'pick'")
    parser.add_argument("--simulate", action="store_true",
                        help="rendered photographed-document mode")
    parser.add_argument("--output", default=None, help="output video path")
    parser.add_argument("--duration", type=float, default=28,
                        help="simulation duration in seconds (default 28)")
    parser.add_argument("--no-display", action="store_true",
                        help="no preview window (headless render)")
    args = parser.parse_args(argv)

    if args.simulate:
        source = None
    elif args.source in (None, "pick"):
        source = pick_video_file()
        if source is None:
            print("No video chosen. Use --simulate for the rendered demo, "
                  "or --source <video> / --source 0 for webcam.")
            return 2
    elif args.source.isdigit():
        source = int(args.source)
    else:
        source = args.source

    stats = run(source, args.simulate, args.output, args.duration,
                args.no_display)
    return 0 if stats is not None else 1


if __name__ == "__main__":
    raise SystemExit(main())
