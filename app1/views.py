#django.
from django.shortcuts import render
from django.http import HttpResponse, JsonResponse, FileResponse
from django.http import JsonResponse
#django restframwork.
from rest_framework.views import APIView
from rest_framework.response import Response
#python libraries.
from sklearn.decomposition import PCA
import pandas as pd
import scipy as sp
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure
#serializers.
from .serializer import (VectorOperationSerializer, MatrixOperationSerializer,
                         LinearSerializer, SVD, LinearRegression, EigenValueSpace,
                         MatrixDecomposition, PCASerializer)


# problem - 1 view
class VectorOperationView(APIView):
    serializer_class = VectorOperationSerializer

    def get_serializer(self, *args, **kwargs):
        return VectorOperationSerializer(*args, **kwargs)

    def post(self, request):      
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        vectorA = serializer.validated_data['vectorA'].strip().split(",")
        vectorA = [int(x) for x in vectorA]
        vectorB = serializer.validated_data['vectorB'].strip().split(",")
        vectorB = [int(x) for x in vectorB]
        scalar = serializer.validated_data['scalar']
        operator = serializer.validated_data['operator']

        if operator == 'add':
            result = [vectorA[0] + vectorB[0], vectorA[1] + vectorB[1]]
            return Response({'result': result})
        
        elif operator == 'subtract':
            result = [vectorA[0] - vectorB[0], vectorA[1] - vectorB[1]]
            return Response({'result': result})
        
        elif operator == 'multiply':
            result = [scalar*vectorA[0], scalar*vectorA[1]]
            return Response({'result': result})
        
        else:
            return Response({'result': "invalid operation"})
        
        

#problem-2 view
#API for Matrix operation #problem No: ###
#Need to synchronize matrix field according to row and column.
class MatrixOperationView(APIView):
    serializer_class = MatrixOperationSerializer

    def get_serializer(self, *args, **kwargs):
        return MatrixOperationSerializer(*args, **kwargs)
    
    def post(self, request):
        serializer = self.get_serializer(data = request.data)
        serializer.is_valid(raise_exception = True)

        row = serializer.validated_data['row']
        col = serializer.validated_data['col']       
        matrix = np.array(serializer.validated_data['matrix'])
        operation = serializer.validated_data['operation']
        
        if operation == 'row_reduced_form':
            q, r = np.linalg.qr(matrix)
            rrf = np.round(np.array([row / row[row != 0][0] for row in r]))
            rrf = rrf.astype(int)
            return Response({'result':rrf})
            
        elif operation == 'determinant':
            determinant = round(np.linalg.det(matrix))
            return Response({'result':determinant})
        
        elif operation == 'rank':
            rank = round(np.linalg.matrix_rank(matrix))
            return Response({'result':rank})
        
        # elif operation == 'echelon_form':
        #     echelon =[np.linalg.inv(matrix)]
        #     return Response({'result':echelon})
        
        else:
            return Response({'result':"Invalid Operation Choice"})

#problem-3 view
#Linear Equation solver
#Used LU Decomposition method.
#need to synchronize variable count according to eqn_LHS and eqn_RHS 
class LinearView(APIView):
    serializer_class = LinearSerializer

    def get_serializer(self, *args, **kwargs):
        return LinearSerializer(*args, **kwargs)
    
    def post(self, request):
        serializer = self.get_serializer(data = request.data)
        serializer.is_valid(raise_exception=True)

        variable_count = serializer.validated_data['variable_count']
        eqn_LHS = np.array(serializer.validated_data['eqn_LHS'])
        eqn_RHS = np.array(serializer.validated_data['eqn_RHS'])

        result = np.round(np.linalg.solve(eqn_LHS,eqn_RHS))
        result = result.astype(int)
        
        return Response({'result':result})

#Problem -4 image processing.
#.....
#.....
#.....
#.....
#.....


#problem - 5 view
class EigenValueSpaceView(APIView):
    serializer_class = EigenValueSpace

    def get_serializer(self, *args, **kwargs):
        return EigenValueSpace(*args, **kwargs)

    def find_eigen(self,matrix):
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        response = {
            'eigenvalues':(np.round(eigenvalues)).astype(int),
            'eigenvectors':(np.round(eigenvectors)).astype(int),
        }
        return Response(response)
    
    def visualize_eigen_space(self, matrix):

        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        # Extract eigenvalues and eigenvectors
        lambda1, lambda2 = eigenvalues
        v1, v2 = eigenvectors[:, 0], eigenvectors[:, 1]

        # Create grid of points
        x = np.linspace(-1, 1, 10)
        y = np.linspace(-1, 1, 10)
        X, Y = np.meshgrid(x, y)

        # Compute transformed grid
        Xt = lambda1 * X
        Yt = lambda2 * Y

        # Plot original and transformed grid
        plt.figure()
        plt.quiver(0, 0, v1[0], v1[1], angles='xy', scale_units='xy', scale=1, color='r', label='Eigenvector 1')
        plt.quiver(0, 0, v2[0], v2[1], angles='xy', scale_units='xy', scale=1, color='b', label='Eigenvector 2')
        plt.quiver(np.zeros_like(X), np.zeros_like(Y), Xt, Yt, angles='xy', scale_units='xy', scale=1, color='g', alpha=0.3)
        plt.xlim([-1, 1])
        plt.ylim([-1, 1])
        plt.xlabel('x')
        plt.ylabel('y')
        plt.title('Eigen Space Visualization')
        plt.legend()
        plt.grid()

        # Save the plot to a BytesIO stream
        image_stream = io.BytesIO()
        plt.savefig(image_stream, format='jpeg')
        plt.close()
        image_stream.seek(0)

        #Json serialization.
        eigenvalues = eigenvalues.tolist()
        eigenvectors = eigenvectors.tolist()
        image_data = base64.b64encode(image_stream.getvalue()).decode('utf-8')
        response = {
            'eigenvalues':np.round(eigenvalues),
            'eigenvectors':np.round(eigenvectors),
            'image':image_data
        }
        return JsonResponse(response)

    def post(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)
        matrix = np.array(serializer.validated_data['matrix'])
        if matrix.shape == (2,2):
            return self.visualize_eigen_space(matrix)
        else:
            return self.find_eigen(matrix)


#problem -6 view
class MatrixDecompositionView(APIView):
    serializer_class = MatrixDecomposition

    def get_serializer(self,*args, **kwargs):
        return MatrixDecomposition(*args, **kwargs)

    def diagonalize_matrix(self,matrix):
        eigenvalues, eigenvectors = np.linalg.eig(matrix)
        #check if diagonalisable.
        if len(eigenvalues) != len(set(eigenvalues)):
            return "Given square matrix is diagonalizable"
        else:
            diagonal_matrix = np.diag(eigenvalues)
            inverse_eigenvectors = np.linalg.inv(eigenvectors)
            diagonalized_matrix = np.dot(inverse_eigenvectors, np.dot(matrix, eigenvectors))
            return diagonalized_matrix
        
    def lu_decomposition(self,matrix):
        P,L,U = sp.linalg.lu(matrix)
        return P,L,U
    
    def qr_decomposition(self,matrix):
        Q, R = np.linalg.qr(matrix)
        return Q,R
    
    def post(self, request):
        serializer = self.get_serializer(data = request.data)
        serializer.is_valid(raise_exception = True)

        matrix = serializer.validated_data['matrix']

        diag_matrix = self.diagonalize_matrix(matrix)
        P,L,U = self.lu_decomposition(matrix)
        Q,R = self.qr_decomposition(matrix)

        response = {
            'diag_matrix': (np.round(diag_matrix)).astype(int),
            'P': (np.round(P)).astype(int),
            'L': (np.round(L)).astype(int),
            'U': (np.round(U)).astype(int),
            'Q': (np.round(Q)).astype(int),
            'R': (np.round(R)).astype(int),
        }
        return Response(response)


    


    
    #need to complete.
    
#problem - 7 view
class LinearRegressionView(APIView):
    serializer_class = LinearRegression

    def get_serializer(self, *args, **kwargs):
        return LinearRegression(*args, **kwargs)
    

    def linear_regression(self, x, y):
        # Data
        x1 = np.array(x)  # Needed for scatter plot
        X = np.array(x)
        Y = np.array(y)

        # Add a column of ones to X for the intercept term
        X = np.column_stack((np.ones_like(X), X))

        # Calculate the coefficients using linear regression
        beta = np.linalg.inv(X.T @ X) @ X.T @ Y

        # Extract the slope and intercept
        intercept = beta[0]
        slope = beta[1]

        def y_generate(x):
            return slope * x + intercept

        # Plotting
        fig, ax = plt.subplots()
        ax.scatter(x1, Y)
        ax.plot(np.linspace(0, 50, 1000), list(map(y_generate, np.linspace(0, 50, 1000))))
        plt.xlabel('X-axis', color='r', size=12)
        plt.ylabel('Y-axis', color='r', size=12)

        # Add equation text to the plot
        equation_text = f'y = {slope:.2f}x + {intercept:.2f}'
        ax.text(0.69, 0.1, equation_text, transform=ax.transAxes, fontsize=12)

        # Converting the plot to binary data
        image_stream = io.BytesIO()
        plt.savefig(image_stream, format='jpeg')
        plt.close(fig)
        image_stream.seek(0)

        return HttpResponse(image_stream, content_type='image/jpeg')

    
    def post(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        data1 = serializer.validated_data['data1']
        data2 = serializer.validated_data['data2']

        return self.linear_regression(data1, data2)

#Problem -8 PCA Analysis.
class PCAView(APIView):
    serializer_class = PCASerializer

    def get_serializer(self, *args, **kwargs):
        return PCASerializer(*args, **kwargs)

    def pca_visualization(self, data):
        pca = PCA()
        transformed_data = pca.fit_transform(data)

        # Results
        explained_variance = pca.explained_variance_ratio_
        cumulative_variance = np.cumsum(explained_variance)

        threshold = 0.95
        num_components = np.argmax(cumulative_variance >= threshold) + 1
        print("Number of components:", num_components)

        # Outlier detection
        residual_variance = np.sum(pca.explained_variance_[num_components:])
        outliers = np.where(np.abs(transformed_data[:, num_components:]) > 3 * np.sqrt(residual_variance))

        # Plotting
        fig = Figure(figsize=(10, 6))
        canvas = FigureCanvas(fig)
        ax = fig.add_subplot(111)

        ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, 'b-', label='Cumulative Explained Variance')
        ax.axvline(x=num_components, color='r', linestyle='--', label='Threshold')
        ax.scatter(outliers[1] + 1, np.zeros_like(outliers[1]), color='red', marker='x', label='Outliers')
        ax.set_xlabel('Number of Components')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.set_title('PCA - Cumulative Explained Variance')
        ax.legend()

        canvas.draw()

        image_stream = io.BytesIO()
        canvas.print_png(image_stream)
        plt.close(fig)
        image_stream.seek(0)
        # image_stream_data = base64.b64encode(image_stream.getvalue()).decode('utf-8')
        image_data = base64.b64encode(image_stream.getvalue()).decode('utf-8')

        response = {
            'no_of_components':str(num_components),
            'image_data':image_data
        }
        return JsonResponse(response)

    def post(self, request):
        serializer = self.get_serializer(data=request.data)
        serializer.is_valid(raise_exception=True)

        data = serializer.validated_data['matrix']
        return self.pca_visualization(data)




# problem- 9 view
class SVDView(APIView):
    serializer_class = SVD

    def get_serializer(self,*args, **kwargs):
        return SVD(*args, **kwargs)
    
    def svd(self, img, k_value):
        #open image using PILLOW and converting it into numpy array.
        img = Image.open(img)
        img_array = np.array(img)

        #seperate each channels
        red_channel = img_array[:, :, 0]
        green_channel = img_array[:, :, 1]
        blue_channel = img_array[:, :, 2]

        #performing svd on each channels separately.
        U_red, s_red, V_red = np.linalg.svd(red_channel, full_matrices=False)
        U_green, s_green, V_green = np.linalg.svd(green_channel, full_matrices=False)
        U_blue, s_blue, V_blue = np.linalg.svd(blue_channel, full_matrices=False)

        #setting value for selecting singular values from 's' matrix.
        s_red[k_value:] = 0
        s_green[k_value:] = 0
        s_blue[k_value:] = 0

        #dot product of each channels after slicing using 'k_value'
        red_reconstructed = U_red.dot(np.diag(s_red)).dot(V_red)
        green_reconstructed = U_green.dot(np.diag(s_green)).dot(V_green)
        blue_reconstructed = U_blue.dot(np.diag(s_blue)).dot(V_blue)

        #merging of seperated channels after operation SVD.
        img_reconstructed = np.stack((red_reconstructed, green_reconstructed, blue_reconstructed), axis=2)

        #Forms Pillow Image object.
        img_reconstructed = Image.fromarray(img_reconstructed.astype('uint8'))

        #converting image into binary data
        output_stream = io.BytesIO()
        img_reconstructed.save(output_stream, format='jpeg')
        output_stream.seek(0)

        return HttpResponse(output_stream, content_type = 'image/jpeg')


    def post(self, request):
        serializer = self.get_serializer(data = request.data)
        serializer.is_valid(raise_exception = True)

        img = serializer.validated_data['image']
        k_value = serializer.validated_data['k']
        
        return self.svd(img, k_value)






