from django.core.paginator import Paginator
from django.template.loader import render_to_string
from django.test import RequestFactory, SimpleTestCase, TestCase
from django.urls import reverse

from .models import CompoundLibrary


class PaginationComponentTests(SimpleTestCase):
    def setUp(self):
        self.factory = RequestFactory()

    def render_component(self, page_count, current_page=2):
        request = self.factory.get(
            "/compound/1/",
            {
                "plant_page": current_page,
                "sample_page": 3,
                "pubmed_page": 4,
            },
        )
        page_obj = Paginator(
            list(range(page_count * 10)),
            10,
        ).get_page(current_page)
        return render_to_string(
            "web/includes/pagination_controls.html",
            {
                "page_obj": page_obj,
                "page_param": "plant_page",
                "anchor_id": "plant-sources",
                "show_info": True,
            },
            request=request,
        )

    def test_component_preserves_other_pages_and_adds_anchor(self):
        html = self.render_component(5)

        self.assertIn("sample_page=3", html)
        self.assertIn("pubmed_page=4", html)
        self.assertIn("plant_page=3", html)
        self.assertIn("#plant-sources", html)
        self.assertIn('action="/compound/1/#plant-sources"', html)
        self.assertIn('name="sample_page" value="3"', html)
        self.assertIn('name="pubmed_page" value="4"', html)

    def test_component_only_shows_jump_input_after_three_pages(self):
        self.assertNotIn(
            'id="plant-sources-page-jump"',
            self.render_component(3),
        )
        self.assertIn(
            'id="plant-sources-page-jump"',
            self.render_component(4),
        )


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
        self.assertNotContains(response, 'id="search-page-jump"')

    def test_duplicate_spectra_are_grouped_by_compound_name(self):
        response = self.client.get(reverse("search"), {"q": "aspirin"})

        self.assertEqual(response.context["results"].paginator.count, 2)

    def test_nist_20_is_displayed_as_nist(self):
        response = self.client.get(reverse("search"), {"q": "calcium"})

        self.assertContains(response, ">NIST<")
        self.assertNotContains(response, "NIST 20")

    def test_page_jump_is_only_displayed_after_three_pages(self):
        CompoundLibrary.objects.bulk_create([
            CompoundLibrary(
                standard=f"Aspirin result {index}",
                title=f"Aspirin result {index}",
                spectrum_type="standard",
            )
            for index in range(59)
        ])

        response = self.client.get(reverse("search"), {"q": "aspirin"})

        self.assertEqual(response.context["results"].paginator.num_pages, 4)
        self.assertContains(response, 'id="search-page-jump"')
        self.assertContains(response, 'max="4"')

    def test_smiles_is_not_treated_as_a_name_search(self):
        response = self.client.get(reverse("search"), {"q": "CC(=O)OC1"})

        self.assertTemplateUsed(response, "web/search_not_found.html")
