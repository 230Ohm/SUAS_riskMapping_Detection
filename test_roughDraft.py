import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import roughDraft


class FakeCapture:
    def __init__(self, opened):
        self.opened = opened
        self.settings = []

    def isOpened(self):
        return self.opened

    def set(self, prop, value):
        self.settings.append((prop, value))
        return True


class FakeStitcher:
    def stitch(self, images):
        height = min(image.shape[0] for image in images)
        resized = [image[:height] for image in images]
        return cv2.Stitcher_OK, np.hstack(resized)


class RoughDraftTests(unittest.TestCase):
    def test_search_boundaries_match_supplied_coordinates(self):
        self.assertEqual(
            roughDraft.SEARCH_BOUNDARIES["Search Boundary 1"],
            [
                (36.216341, -96.010424),
                (36.21675, -96.00755),
                (36.218054, -96.007835),
                (36.217645, -96.010709),
            ],
        )
        self.assertEqual(
            roughDraft.SEARCH_BOUNDARIES["Search Boundary 2"],
            [
                (36.213469, -96.002635),
                (36.215795, -96.002635),
                (36.215795, -96.004286),
                (36.213469, -96.004286),
            ],
        )

    def test_search_boundaries_are_four_point_closed_areas(self):
        for points in roughDraft.SEARCH_BOUNDARIES.values():
            self.assertEqual(len(points), 4)
            latitudes = [point[0] for point in points]
            longitudes = [point[1] for point in points]
            self.assertTrue(all(36.21 < latitude < 36.22 for latitude in latitudes))
            self.assertTrue(all(-96.02 < longitude < -96.00 for longitude in longitudes))
            self.assertGreater(self._shoelace_area(points), 0)

    def test_utc_now_iso_returns_utc_timestamp(self):
        timestamp = roughDraft.utc_now_iso()
        self.assertIn("+00:00", timestamp)

    def test_read_demo_gps_returns_empty_placeholder_values(self):
        self.assertEqual(roughDraft.read_demo_gps(), (None, None, None, None))

    def test_ensure_output_dirs_creates_image_directory(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            with mock.patch.object(roughDraft, "IMAGE_DIR", Path(temp_dir) / "images"):
                roughDraft.ensure_output_dirs()
                self.assertTrue(roughDraft.IMAGE_DIR.exists())

    def test_append_metadata_creates_csv_header_and_row(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            metadata_file = Path(temp_dir) / "metadata.csv"
            row = roughDraft.ImageMetadata(
                filename="map_img_00001.jpg",
                timestamp_utc="2026-05-12T15:00:00.000+00:00",
                latitude=36.216341,
                longitude=-96.010424,
                altitude_m=80.0,
                heading_deg=90.0,
                focus=0,
                exposure=-6,
            )

            with mock.patch.object(roughDraft, "METADATA_FILE", metadata_file):
                roughDraft.append_metadata(row)

            with metadata_file.open(newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["filename"], "map_img_00001.jpg")
            self.assertEqual(rows[0]["latitude"], "36.216341")
            self.assertEqual(rows[0]["longitude"], "-96.010424")

    def test_open_camera_uses_directshow_fallback_when_msmf_fails(self):
        msmf_capture = FakeCapture(opened=False)
        dshow_capture = FakeCapture(opened=True)

        with mock.patch.object(roughDraft.cv2, "VideoCapture", side_effect=[msmf_capture, dshow_capture]):
            cap = roughDraft.open_camera(camera_index=1, width=1280, height=720)

        self.assertIs(cap, dshow_capture)
        self.assertIn((cv2.CAP_PROP_FRAME_WIDTH, 1280), dshow_capture.settings)
        self.assertIn((cv2.CAP_PROP_FRAME_HEIGHT, 720), dshow_capture.settings)

    def test_open_camera_raises_when_all_backends_fail(self):
        with mock.patch.object(roughDraft.cv2, "VideoCapture", side_effect=[FakeCapture(False), FakeCapture(False)]):
            with self.assertRaises(RuntimeError):
                roughDraft.open_camera(camera_index=99, width=1280, height=720)

    def test_configure_camera_applies_fixed_mapping_settings(self):
        cap = FakeCapture(opened=True)
        roughDraft.configure_camera(cap, focus=10, exposure=-7)

        self.assertIn((cv2.CAP_PROP_AUTOFOCUS, 0), cap.settings)
        self.assertIn((cv2.CAP_PROP_AUTO_EXPOSURE, 0.25), cap.settings)
        self.assertIn((cv2.CAP_PROP_FOCUS, 10), cap.settings)
        self.assertIn((cv2.CAP_PROP_EXPOSURE, -7), cap.settings)

    def test_stitch_preview_writes_output_file(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir) / "images"
            output_file = Path(temp_dir) / "stitched" / "Team_risk_map_demo.jpg"
            image_dir.mkdir()

            cv2.imwrite(str(image_dir / "map_img_00001.jpg"), self._test_image((255, 0, 0)))
            cv2.imwrite(str(image_dir / "map_img_00002.jpg"), self._test_image((0, 255, 0)))

            with mock.patch.object(roughDraft.cv2, "Stitcher_create", return_value=FakeStitcher()):
                roughDraft.stitch_preview(image_dir, output_file)

            self.assertTrue(output_file.exists())
            stitched = cv2.imread(str(output_file))
            self.assertIsNotNone(stitched)
            self.assertGreater(stitched.shape[1], stitched.shape[0])

    def test_stitch_preview_does_not_write_with_fewer_than_two_images(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_dir = Path(temp_dir) / "images"
            output_file = Path(temp_dir) / "stitched.jpg"
            image_dir.mkdir()

            cv2.imwrite(str(image_dir / "map_img_00001.jpg"), self._test_image((255, 0, 0)))
            roughDraft.stitch_preview(image_dir, output_file)

            self.assertFalse(output_file.exists())

    def test_parse_args_defaults_and_overrides(self):
        with mock.patch.object(sys, "argv", ["roughDraft.py"]):
            args = roughDraft.parse_args()
            self.assertEqual(args.camera_index, 1)
            self.assertEqual(args.width, 1920)
            self.assertFalse(args.stitch_only)

        with mock.patch.object(
            sys,
            "argv",
            ["roughDraft.py", "--camera-index", "2", "--width", "640", "--stitch-only"],
        ):
            args = roughDraft.parse_args()
            self.assertEqual(args.camera_index, 2)
            self.assertEqual(args.width, 640)
            self.assertTrue(args.stitch_only)

    def test_main_stitch_only_calls_stitch_preview(self):
        with mock.patch.object(sys, "argv", ["roughDraft.py", "--stitch-only"]):
            with mock.patch.object(roughDraft, "stitch_preview") as stitch_preview:
                with mock.patch.object(roughDraft, "capture_mapping_images") as capture:
                    roughDraft.main()

        stitch_preview.assert_called_once_with(roughDraft.IMAGE_DIR, roughDraft.FINAL_MAP_FILE)
        capture.assert_not_called()

    @staticmethod
    def _test_image(color):
        image = np.zeros((80, 100, 3), dtype=np.uint8)
        image[:, :] = color
        cv2.putText(image, "SUAS", (15, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return image

    @staticmethod
    def _shoelace_area(points):
        total = 0.0
        for index, (lat_a, lon_a) in enumerate(points):
            lat_b, lon_b = points[(index + 1) % len(points)]
            total += lon_a * lat_b - lon_b * lat_a
        return abs(total) / 2.0


if __name__ == "__main__":
    unittest.main()
