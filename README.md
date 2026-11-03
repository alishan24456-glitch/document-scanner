# Document Scanner

Edge detection + perspective-transform document scanning, built with
OpenCV: the pipeline finds a document's four corners in a frame,
warps it to a flat page, and enhances it into a clean black-on-white
scan — the same steps a phone scanner app uses. A stable detection
auto-captures a scan (first capture after 45 steady frames, then
roughly every 8 seconds while the document stays in view).

## Run it (CMD)

```
pip install -r requirements.txt
python document_scanner.py --simulate
python document_scanner.py --simulate --output demo.mp4 --duration 28
python document_scanner.py --source pick
python document_scanner.py
python document_scanner.py --source 0
python document_scanner.py --source receipt_photo.mp4 --output scanned.mp4
```

Running with no arguments (or `--source pick`) opens a file-picker
dialog to choose a video. Use `--no-display` for a headless render
that only writes `--output`.

## How it works

One pipeline serves both modes: grayscale → blur → Canny edges →
largest 4-point contour → perspective warp to a 500×700 page →
adaptive-threshold enhancement → overlay (corner markers, scanned
inset, live HUD, SCANNED banner on each auto-capture).

- **Real mode** runs that detector on a video file or webcam feed.
  It needs a document with a visible border against a contrasting
  background; glare, a curled page edge, or a background the same
  colour as the paper can break the contour.
- **Simulation mode** renders a skewed "photographed" report on a desk
  (drifting and tilting gently over time) and reads it back out of
  the rendered pixels with the same detector. It never reads the
  scene's corner positions: detection comes only from pixels, exactly
  as real-mode detection comes only from camera pixels. Simulation
  needs only opencv-python + numpy.

## Demo

- Simulation demo (28 s, rendered desk scene, judged by the pixel
  detector): `files/project-demos/document-scanner-demo.mp4`
- LinkedIn-ready copy (H.264, same demo):
  `files/project-demos/linkedin-h264/document-scanner-demo.mp4`
- Verification: `py_compile` clean; demo read back and frame-inspected;
  end-of-run counters below are the run's own output, not estimates.
  - Simulation demo (1 Nov 2026): 840 frames at 30 fps, 840 with the
    document detected (100%), 4 scans auto-captured, longest stable
    detection streak 840 frames, ends DOCUMENT DETECTED.

## Honest notes

- The demo shows the pipeline working on a clean rendered scene with
  strong paper/desk contrast; real photos vary with lighting, glare,
  shadows and page curl, and the detector will miss pages whose
  border is not the dominant quadrilateral in the frame.
- The "scan" is an enhanced thresholded image, not OCR: it cleans up
  the page for reading or archiving but extracts no text. Adding
  OCR (e.g. Tesseract) is a natural next step.
- Detection is purely geometric — it finds any quadrilateral page
  shape, so a book, a laptop screen or a picture frame in view can be
  picked up instead of the intended document.
- The simulation demo shows the pipeline working; on real footage
  the same detector is the mode that matters, and its accuracy
  depends on contrast, lighting and camera angle.
