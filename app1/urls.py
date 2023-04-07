from django.urls import path
from .views import VectorOperationView

urlpatterns = [
    path('vector/', VectorOperationView.as_view()),
]

