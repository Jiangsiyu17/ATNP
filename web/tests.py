from django.test import TestCase
from django.urls import reverse

from .models import CompoundLibrary


class CompoundNameSearchTests(TestCase):
    def setUp(self):
        self.aspirin = CompoundLibrary.objects.create(
            standard="Aspirin",
            title="Aspirin spectrum",
            smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
            database="ATNP",
            ionmode="negative",
            spectrum_type="standard",
        )
        CompoundLibrary.objects.create(
            standard="Aspirin",
            title="Aspirin positive spectrum",
            smiles="CC(=O)OC1=CC=CC=C1C(=O)O",
            database="GNPS",
            ionmode="positive",
            spectrum_type="standard",
        )
        self.related = CompoundLibrary.objects.create(
            standard="Aspirin calcium",
            title="Aspirin calcium spectrum",
            smiles="CC(=O)OC1=CC=CC=C1C(=O)[O-]",
            database="NIST 20",
            ionmode="negative",
            spectrum_type="standard",
        )

    def test_search_displays_similar_named_compounds_before_detail(self):
        response = self.client.get(reverse("search"), {"q": "aspirin"})

        self.assertEqual(response.status_code, 200)
        self.assertTemplateUsed(response, "web/search_results.html")
        self.assertContains(response, "Aspirin")
        self.assertContains(response, "Aspirin calcium")
        self.assertContains(
            response,
            reverse("compound_detail", args=[self.aspirin.pk]),
        )

    def test_duplicate_spectra_are_grouped_by_compound_name(self):
        response = self.client.get(reverse("search"), {"q": "aspirin"})

        self.assertEqual(response.context["results"].paginator.count, 2)

    def test_nist_20_is_displayed_as_nist(self):
        response = self.client.get(reverse("search"), {"q": "calcium"})

        self.assertContains(response, ">NIST<")
        self.assertNotContains(response, "NIST 20")

    def test_smiles_is_not_treated_as_a_name_search(self):
        response = self.client.get(reverse("search"), {"q": "CC(=O)OC1"})

        self.assertTemplateUsed(response, "web/search_not_found.html")
