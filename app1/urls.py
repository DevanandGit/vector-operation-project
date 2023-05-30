from django.urls import path
from .views import (VectorOperationView, MatrixOperationView,
                    LinearView, SVDView, LinearRegressionView,
                    EigenValueSpaceView, MatrixDecompositionView,
                    PCAView)

urlpatterns = [
    
    path('matrix/', MatrixOperationView.as_view(), name='matrix'), #problem - 1
    path('lineqn/', LinearView.as_view(), name='lineqn'),          #problem-2
    path('vector/', VectorOperationView.as_view(), name='vector'), #problem-3
    # path('imagepro', LinearTransformationView.as_view(), name='imagepro') #problem -4
    path('eigen/', EigenValueSpaceView.as_view(), name="eigen"),#problem - 5
    path('matdecomp/', MatrixDecompositionView.as_view(), name='matdecomp'),#problem -6
    path('lr/', LinearRegressionView.as_view(), name = 'LinearRegression'),#problem - 7
    path('pca/', PCAView.as_view(), name='pca'),#Problem -8
    path('svd', SVDView.as_view(), name='svd'),#problem-9    
]

