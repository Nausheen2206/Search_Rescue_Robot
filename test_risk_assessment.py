import unittest

from risk_assessment import Detection, RiskAssessor


def detection(class_name: str, confidence: float = 0.9) -> Detection:
    return Detection(class_name, confidence, (10, 10, 60, 60))


class RiskAssessorTests(unittest.TestCase):
    def test_single_fire_detection_is_candidate_not_confirmed(self) -> None:
        assessor = RiskAssessor()

        result = assessor.update([detection("Fire")], 100, 100)

        self.assertEqual(result.hazard_level, "LOW")
        self.assertEqual(result.candidates, ("Fire",))
        self.assertNotIn("Fire", result.confirmed_classes)

    def test_fire_requires_three_of_five_frames_and_sets_high_hazard(self) -> None:
        assessor = RiskAssessor()

        assessor.update([detection("Fire")], 100, 100)
        assessor.update([], 100, 100)
        result = assessor.update([detection("Fire")], 100, 100)
        self.assertEqual(result.hazard_level, "LOW")

        result = assessor.update([detection("Fire")], 100, 100)

        self.assertEqual(result.hazard_level, "HIGH")
        self.assertIn("Fire", result.confirmed_classes)
        self.assertEqual(result.candidates, ())

    def test_smoke_is_moderate_and_person_is_separate_rescue_priority(self) -> None:
        assessor = RiskAssessor()
        result = None

        for _ in range(3):
            result = assessor.update(
                [detection("Smoke"), detection("Person")],
                100,
                100,
            )

        self.assertIsNotNone(result)
        self.assertEqual(result.hazard_level, "MODERATE")
        self.assertEqual(result.rescue_priority, "PERSON PRESENT")
        self.assertIn("Smoke", result.confirmed_classes)
        self.assertIn("Person", result.confirmed_classes)

    def test_confidence_below_threshold_does_not_count(self) -> None:
        assessor = RiskAssessor()

        for _ in range(5):
            result = assessor.update([detection("Fire", 0.49)], 100, 100)

        self.assertEqual(result.hazard_level, "LOW")
        self.assertEqual(result.candidates, ())

    def test_confirmation_expires_when_old_detections_leave_window(self) -> None:
        assessor = RiskAssessor()
        for _ in range(3):
            result = assessor.update([detection("Fire")], 100, 100)
        self.assertEqual(result.hazard_level, "HIGH")

        for _ in range(3):
            result = assessor.update([], 100, 100)

        self.assertEqual(result.hazard_level, "LOW")
        self.assertNotIn("Fire", result.confirmed_classes)

    def test_detection_context_reports_size_and_position(self) -> None:
        assessor = RiskAssessor()

        result = assessor.update(
            [Detection("Person", 0.9, (0, 0, 80, 80))],
            100,
            100,
        )

        self.assertEqual(result.detections[0].relative_size, "large")
        self.assertEqual(result.detections[0].position, "center/middle")

    def test_invalid_window_configuration_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            RiskAssessor(window_frames=2, confirm_frames=3)


if __name__ == "__main__":
    unittest.main()
