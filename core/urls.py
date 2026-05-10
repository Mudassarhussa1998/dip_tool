from django.urls import path
from . import views

urlpatterns = [
    # ── Page routes ──────────────────────────────────────────────────────────
    path('', views.home, name='home'),
    path('intro/', views.intro, name='intro'),
    path('sampling/', views.sampling, name='sampling'),
    path('enhancement/', views.enhancement, name='enhancement'),
    path('bitplane/', views.bitplane, name='bitplane'),
    path('histogram/', views.histogram_page, name='histogram'),
    path('spatial/', views.spatial, name='spatial'),
    path('sharpening/', views.sharpening, name='sharpening'),
    path('frequency/', views.frequency_page, name='frequency'),
    path('restoration/', views.restoration, name='restoration'),
    path('segmentation-edges/', views.segmentation_edges, name='segmentation_edges'),
    path('segmentation-region/', views.segmentation_region, name='segmentation_region'),
    path('morphology/', views.morphology_page, name='morphology'),
    path('compression/', views.compression_page, name='compression'),
    path('colour/', views.colour, name='colour'),

    # ── API routes ────────────────────────────────────────────────────────────
    path('api/intro/', views.api_intro, name='api_intro'),
    path('api/sampling/', views.api_sampling, name='api_sampling'),
    path('api/enhancement/', views.api_enhancement, name='api_enhancement'),
    path('api/bitplane/', views.api_bitplane, name='api_bitplane'),
    path('api/histogram/', views.api_histogram, name='api_histogram'),
    path('api/spatial/', views.api_spatial, name='api_spatial'),
    path('api/sharpening/', views.api_sharpening, name='api_sharpening'),
    path('api/frequency/', views.api_frequency, name='api_frequency'),
    path('api/restoration/', views.api_restoration, name='api_restoration'),
    path('api/segmentation-edges/', views.api_segmentation_edges, name='api_segmentation_edges'),
    path('api/segmentation-region/', views.api_segmentation_region, name='api_segmentation_region'),
    path('api/morphology/', views.api_morphology, name='api_morphology'),
    path('api/compression/', views.api_compression, name='api_compression'),
    path('api/colour/', views.api_colour, name='api_colour'),
]
