from django.urls import path
from .views import VectorOperationView, MatrixOperationView,LinearView

urlpatterns = [
    path('vector/', VectorOperationView.as_view(), name='vector'),
    path('matrix/', MatrixOperationView.as_view(), name='matrix'),
    path('lineqn/', LinearView.as_view(), name='lineqn')
]

