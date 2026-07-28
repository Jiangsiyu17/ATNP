from django.urls import path, re_path
from . import views

urlpatterns = [
    path('compound/list/', views.compound_list, name='compound_list'),
    path('compound/<int:pk>/', views.compound_detail, name='compound_detail'),
    path("compounds/api/", views.compound_list_api, name="compound_list_api"),


    path('plants/', views.plant_list, name='plant_list'),
    path("plants/api/", views.plant_list_api, name="plant_list_api"),
    path('plant/<str:latin_name>/', views.plant_detail, name='plant_detail'),
    path("api/plant/<str:latin_name>/", views.plant_detail_api),

    path('', views.home, name='home'),
    path('search/', views.search, name='search'),

    re_path(r"^plant/(?P<latin_name>[^/]+)/(?P<compound_id>[^/]+)/$",
            views.plant_compound_detail, name="plant_compound_detail"),

    path("similar/<int:compound_id>/<int:spectrum_idx>/",
         views.similar_compare, name="similar_compare"),

    # Structure Query
    path("structure-query/", views.structure_query, name="structure_query"),
    path("structure-query/result/", views.structure_search, name="structure_result"),

    path('molecular_weight_query/', views.molecular_weight_query, name='molecular_weight_query'),
    path('molecular_weight_search/', views.molecular_weight_search, name='molecular_weight_search'),
    path("mw/api/", views.mw_api, name="mw_api"),

    path('msms_search/', views.msms_search, name='msms_search'),
    path('msms_result/', views.msms_result, name='msms_result'),

]
