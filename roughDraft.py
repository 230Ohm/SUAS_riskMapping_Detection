from __future__ import annotations
import argparse
import csv
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
import cv2


TEAM_NAME = "riverCityRockets"
OUTPUT_DIR = Path("suas_mapping_demo")
IMAGE_DIR = OUTPUT_DIR / "mission_images"
METADATA_FILE = OUTPUT_DIR / "metadata.csv"
FINAL_MAP_FILE = OUTPUT_DIR / f"{TEAM_NAME}_riskMapDemo.jpg"

SEARCH_BOUNDARIES = {
    "Search Boundary 1": [
        (36.216341, -96.010424),
        (36.21675, -96.00755),
        (36.218054, -96.007835),
        (36.217645, -96.010709),
    ],
    "Search Boundary 2": [
        (36.213469, -96.002635),
        (36.215795, -96.002635),
        (36.215795, -96.004286),
        (36.213469, -96.004286),
    ],
}


@dataclass
class ImageMetadata:
    filename: str
    timestamp_utc: str
    latitude: Optional[float]
    longitude: Optional[float]
    altitude_m: Optional[float]
    heading_deg: Optional[float]
    focus: int
    exposure: int


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


def ensure_output_dirs() -> None:
    IMAGE_DIR.mkdir(parents=True, exist_ok=True)


def append_metadata(row: ImageMetadata) -> None:
    is_new_file = not METADATA_FILE.exists()

    with METADATA_FILE.open("a", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=asdict(row).keys())
        if is_new_file:
            writer.writeheader()
        writer.writerow(asdict(row))


def read_demo_gps() -> tuple[Optional[float], Optional[float], Optional[float], Optional[float]]:
    """
    Placeholder for real telemetry.

    Replace this with MAVLink, DroneKit, ROS, serial GPS, or autopilot SDK reads.
    Returning None is fine for camera testing, but real map production should log
    GPS/altitude/heading for every image.
    """

    latitude = None
    longitude = None
    altitude_m = None
    heading_deg = None
    return latitude, longitude, altitude_m, heading_deg


def open_camera(camera_index: int, width: int, height: int) -> cv2.VideoCapture:
    cap = cv2.VideoCapture(camera_index, cv2.CAP_MSMF)

    if not cap.isOpened():
        print("MSMF camera open failed; trying DirectShow fallback...")
        cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)

    if not cap.isOpened():
        raise RuntimeError(f"Could not open camera index {camera_index}")

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    return cap


def configure_camera(cap: cv2.VideoCapture, focus: int, exposure: int) -> None:
    # Fixed settings reduce exposure/focus changes between frames, which helps stitching.
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)
    cap.set(cv2.CAP_PROP_FOCUS, focus)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)


def capture_mapping_images(args: argparse.Namespace) -> None:
    ensure_output_dirs()

    cap = open_camera(args.camera_index, args.width, args.height)
    configure_camera(cap, args.focus, args.exposure)

    print("\n--- SUAS Mapping Capture Demo ---")
    print(f"Images:   {IMAGE_DIR.resolve()}")
    print(f"Metadata: {METADATA_FILE.resolve()}")
    print("Q: quit | SPACE: save immediate frame\n")

    image_count = 0
    last_capture_time = time.time()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                print("Camera frame read failed.")
                break

            elapsed = time.time() - last_capture_time
            should_capture = elapsed >= args.interval_seconds

            display = frame.copy()
            status = (
                f"Team: {TEAM_NAME} | Images: {image_count} | "
                f"F:{args.focus} E:{args.exposure} | Next: {max(0, args.interval_seconds - elapsed):.1f}s"
            )
            cv2.putText(display, status, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
            cv2.imshow("SUAS Mapping Capture", display)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == ord(" "):
                should_capture = True

            if should_capture:
                filename = f"map_img_{image_count:05d}.jpg"
                image_path = IMAGE_DIR / filename

                cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality])
                latitude, longitude, altitude_m, heading_deg = read_demo_gps()

                append_metadata(
                    ImageMetadata(
                        filename=filename,
                        timestamp_utc=utc_now_iso(),
                        latitude=latitude,
                        longitude=longitude,
                        altitude_m=altitude_m,
                        heading_deg=heading_deg,
                        focus=args.focus,
                        exposure=args.exposure,
                    )
                )

                print(f"[saved] {image_path} | metadata appended")
                image_count += 1
                last_capture_time = time.time()

    finally:
        cap.release()
        cv2.destroyAllWindows()


def stitch_preview(image_folder: Path, output_file: Path) -> None:
    image_paths = sorted(image_folder.glob("*.jpg")) + sorted(image_folder.glob("*.png"))
    images = []

    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            print(f"[skip] Could not read {path}")
            continue
        images.append(image)

    if len(images) < 2:
        print("Need at least two images to stitch.")
        return

    print(f"Attempting OpenCV stitch with {len(images)} images...")
    stitcher = cv2.Stitcher_create(cv2.Stitcher_PANORAMA)
    status, stitched = stitcher.stitch(images)

    if status != cv2.Stitcher_OK:
        print(f"OpenCV stitch failed with status {status}.")
        print("This is common for nadir mapping. Use WebODM/OpenDroneMap for the real orthomosaic.")
        return

    output_file.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_file), stitched, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[map preview saved] {output_file.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SUAS imagery mapping capture demo")
    parser.add_argument("--camera-index", type=int, default=1)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--interval-seconds", type=float, default=5.0)
    parser.add_argument("--focus", type=int, default=0) 
    parser.add_argument("--exposure", type=int, default=-6)
    parser.add_argument("--jpeg-quality", type=int, default=95)
    parser.add_argument("--stitch-only", action="store_true", help="Only stitch existing images in the output folder")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.stitch_only:
        stitch_preview(IMAGE_DIR, FINAL_MAP_FILE)
        return

    capture_mapping_images(args)
    stitch_preview(IMAGE_DIR, FINAL_MAP_FILE)


if __name__ == "__main__":
    main()
