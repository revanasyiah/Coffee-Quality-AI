"""Pengujian alur input -> proses -> output. Jalankan: python manage.py test"""
from types import SimpleNamespace
from unittest.mock import patch

from django.test import TestCase, override_settings
from django.urls import reverse
from google.genai.errors import APIError

from .models import PredictionHistory
from .services import GeminiAnalysisError, generate_ai_analysis, get_metadata, predict_score


def valid_payload(**over):
    ch = get_metadata()["choices"]
    data = {"altitude": 1300, "moisture": 10.8, "category_one_defects": 0,
            "category_two_defects": 2, "quakers": 0,
            "country": ch["Country of Origin"][0], "variety": ch["Variety"][0],
            "processing_method": ch["Processing Method"][0], "color": ch["Color"][0]}
    data.update(over)
    return data


class ModelServiceTests(TestCase):
    def test_model_loads_and_predicts_in_valid_range(self):
        ch = get_metadata()["choices"]
        score = predict_score({"Altitude": 1300, "Moisture Percentage": 10.8,
                               "Category One Defects": 0, "Category Two Defects": 2, "Quakers": 0,
                               "Country of Origin": "Colombia", "Variety": "Caturra",
                               "Processing Method": "Washed / Wet", "Color": "green"})
        self.assertTrue(70 < score < 95)

    def test_unseen_category_does_not_crash(self):
        score = predict_score({"Altitude": 1300, "Moisture Percentage": 10.8,
                               "Category One Defects": 0, "Category Two Defects": 0, "Quakers": 0,
                               "Country of Origin": "Atlantis", "Variety": "XYZ",
                               "Processing Method": "Washed / Wet", "Color": "green"})
        self.assertTrue(70 < score < 95)

    def test_no_leakage_features(self):
        meta = get_metadata()
        self.assertFalse(set(meta["features"]) & set(meta["excluded_leakage_columns"]))

    @override_settings(
        GEMINI_API_KEY="test-key",
        GEMINI_MODEL="primary-model",
        GEMINI_FALLBACK_MODEL="fallback-model",
    )
    @patch("coffee_app.services.time.sleep")
    @patch("google.genai.Client")
    def test_gemini_overload_retries_then_uses_fallback(
            self, client_class, sleep):
        client = client_class.return_value
        overloaded = APIError(503, {
            "error": {"code": 503, "message": "busy", "status": "UNAVAILABLE"}})
        client.models.generate_content.side_effect = [
            overloaded, overloaded, SimpleNamespace(text="Analisis berhasil.")
        ]

        result = generate_ai_analysis({
            "predicted_score": 84.2, "category": "Very Good", "country": "Colombia",
            "variety": "Caturra", "processing_method": "Washed / Wet", "color": "green",
            "altitude": 1300, "moisture": 10.8, "category_one_defects": 0,
            "category_two_defects": 2, "quakers": 0,
        })

        self.assertEqual(result, "Analisis berhasil.")
        self.assertEqual(
            [call.kwargs["model"] for call in client.models.generate_content.call_args_list],
            ["primary-model", "primary-model", "fallback-model"],
        )
        sleep.assert_called_once_with(1)


class ViewTests(TestCase):
    def test_pages_load(self):
        for name in ["dashboard", "predict", "insights", "history"]:
            self.assertEqual(self.client.get(reverse(f"coffee_app:{name}")).status_code, 200, name)

    def test_dashboard_has_gemini_buttons_for_both_charts(self):
        response = self.client.get(reverse("coffee_app:dashboard"))
        self.assertContains(response, "Analisis distribusi kategori dengan Gemini")
        self.assertContains(response, "Analisis distribusi skor dengan Gemini")

    @override_settings(
        GEMINI_API_KEY="test-key",
        GEMINI_MODEL="primary-model",
        GEMINI_FALLBACK_MODEL="",
    )
    @patch("google.genai.Client")
    def test_dashboard_chart_buttons_send_each_chart_to_gemini(self, client_class):
        client = client_class.return_value
        client.models.generate_content.side_effect = [
            SimpleNamespace(text="Analisis kategori dari Gemini."),
            SimpleNamespace(text="Analisis histogram dari Gemini."),
        ]

        for chart_type, expected in [
                ("categories", "Analisis kategori dari Gemini."),
                ("total_points", "Analisis histogram dari Gemini.")]:
            response = self.client.post(
                reverse("coffee_app:analyze_dashboard_chart", args=[chart_type]))
            self.assertEqual(response.status_code, 200)
            self.assertContains(response, expected)

        calls = client.models.generate_content.call_args_list
        self.assertEqual([call.kwargs["model"] for call in calls], ["primary-model", "primary-model"])
        self.assertIn("Jumlah skor di bawah 80", calls[0].kwargs["contents"])
        self.assertIn("Rata-rata:", calls[1].kwargs["contents"])

    @override_settings(GEMINI_API_KEY="")
    def test_dashboard_chart_shows_missing_api_key_error(self):
        response = self.client.post(
            reverse("coffee_app:analyze_dashboard_chart", args=["categories"]))
        self.assertContains(response, "API key Gemini belum diatur")
        self.assertContains(response, "Analisis distribusi kategori dengan Gemini")

    def test_insights_has_gemini_buttons_for_all_requested_sections(self):
        response = self.client.get(reverse("coffee_app:insights"))
        self.assertContains(response, 'class="gemini-insight-form mt-3"', count=8)

    @override_settings(
        GEMINI_API_KEY="test-key",
        GEMINI_MODEL="primary-model",
        GEMINI_FALLBACK_MODEL="",
    )
    @patch("google.genai.Client")
    def test_insight_buttons_send_each_section_to_gemini(self, client_class):
        client = client_class.return_value
        client.models.generate_content.side_effect = [
            SimpleNamespace(text=f"Analisis Gemini {index}.") for index in range(8)
        ]
        sections = [
            "algorithm_comparison", "model_comparison", "actual_predicted",
            "feature_importance", "numeric_correlation", "country_average",
            "processing_method", "sensory_leakage",
        ]

        for index, section in enumerate(sections):
            response = self.client.post(
                reverse("coffee_app:analyze_insight", args=[section]))
            self.assertEqual(response.status_code, 200, section)
            self.assertContains(response, f"Analisis Gemini {index}.")

        calls = client.models.generate_content.call_args_list
        self.assertEqual(len(calls), 8)
        prompts = [call.kwargs["contents"] for call in calls]
        self.assertIn("Linear Regression", prompts[0])
        self.assertIn("RMSE CV", prompts[1])
        self.assertIn("RMSE uji", prompts[2])
        self.assertIn("Variety", prompts[3])
        self.assertIn("Korelasi Pearson", prompts[4])
        self.assertIn("rata-rata", prompts[5])
        self.assertIn("Washed / Wet", prompts[6])
        self.assertIn("data leakage", prompts[7])

    @override_settings(GEMINI_API_KEY="")
    def test_insight_chart_shows_missing_api_key_error(self):
        response = self.client.post(
            reverse("coffee_app:analyze_insight", args=["algorithm_comparison"]))
        self.assertContains(response, "API key Gemini belum diatur")
        self.assertContains(response, "Analisis perbandingan algoritma dengan Gemini")

    def test_full_flow_saves_history_and_shows_result(self):
        r = self.client.post(reverse("coffee_app:predict"), valid_payload(), follow=True)
        self.assertEqual(r.status_code, 200)
        self.assertEqual(PredictionHistory.objects.count(), 1)
        self.assertContains(r, "Estimasi Total Cup Points")
        self.assertContains(r, "bukan skor cupping resmi")
        self.assertContains(self.client.get(reverse("coffee_app:history")), "Detail")

    def test_validation_rejects_bad_input(self):
        for bad in [{"altitude": 99999}, {"moisture": -5}, {"quakers": -1}, {"altitude": ""}, {"country": "Atlantis"}]:
            r = self.client.post(reverse("coffee_app:predict"), valid_payload(**bad))
            self.assertEqual(r.status_code, 200, bad)  # form ditampilkan ulang dgn error
            self.assertEqual(PredictionHistory.objects.count(), 0, bad)

    def test_out_of_range_input_warns_but_works(self):
        r = self.client.post(reverse("coffee_app:predict"), valid_payload(category_two_defects=90), follow=True)
        self.assertContains(r, "di luar rentang data latih")
        self.assertEqual(PredictionHistory.objects.count(), 1)

    def test_result_404_for_unknown_id(self):
        self.assertEqual(self.client.get(reverse("coffee_app:result", args=[999])).status_code, 404)

    @patch("coffee_app.views.generate_ai_analysis", return_value="Analisis singkat dari Gemini.")
    def test_ai_analysis_is_shown_on_result(self, generate_analysis):
        rec = PredictionHistory.objects.create(
            altitude=1300, moisture=10.8, category_one_defects=0, category_two_defects=2,
            quakers=0, country="Colombia", variety="Caturra", processing_method="Washed / Wet",
            color="green", predicted_score=84.2, category="Very Good")
        response = self.client.post(reverse("coffee_app:analyze_result", args=[rec.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Analisis singkat dari Gemini.")
        generate_analysis.assert_called_once()

    @patch("coffee_app.views.generate_ai_analysis", side_effect=GeminiAnalysisError("Gemini sedang sibuk."))
    def test_ai_analysis_error_is_shown_in_analysis_card(self, generate_analysis):
        rec = PredictionHistory.objects.create(
            altitude=1300, moisture=10.8, category_one_defects=0, category_two_defects=2,
            quakers=0, country="Colombia", variety="Caturra", processing_method="Washed / Wet",
            color="green", predicted_score=84.2, category="Very Good")
        response = self.client.post(reverse("coffee_app:analyze_result", args=[rec.pk]))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Gemini sedang sibuk.")
        self.assertContains(response, "Buat analisis dengan Gemini")
        generate_analysis.assert_called_once()

    @override_settings(GEMINI_API_KEY="")
    def test_ai_analysis_without_api_key_shows_configuration_error(self):
        rec = PredictionHistory.objects.create(
            altitude=1300, moisture=10.8, category_one_defects=0, category_two_defects=2,
            quakers=0, country="Colombia", variety="Caturra", processing_method="Washed / Wet",
            color="green", predicted_score=84.2, category="Very Good")
        response = self.client.post(
            reverse("coffee_app:analyze_result", args=[rec.pk]), follow=True)
        self.assertContains(response, "Isi GEMINI_API_KEY pada file .env")
