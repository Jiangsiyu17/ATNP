from django.urls import path, re_path
from . import views

urlpatterns = [
    path('compound/list/', views.compound_list, name='compound_list'),
    path('compound/<int:pk>/', views.compound_detail, name='compound_detail'),

    path('herbs/', views.herb_list, name='herb_list'),
    path('herb/<str:latin_name>/', views.herb_detail, name='herb_detail'),

    path('', views.home, name='home'),
    path('search/', views.search, name='search'),

    re_path(r"^herb/(?P<latin_name>[^/]+)/(?P<compound>[^/]+)/$",
            views.herb_compound_detail, name="herb_compound_detail"),

    path("similar/<int:compound_id>/<int:spectrum_idx>/",
         views.similar_compare, name="similar_compare"),

    # Structure Query
    path("structure-query/", views.structure_query, name="structure_query"),
    path("structure-query/result/", views.structure_search, name="structure_query_result"),


]
