from django.shortcuts import render
from rest_framework.views import APIView
from .serializer import VectorOperationSerializer, MatrixOperationSerializer
from rest_framework.response import Response
import numpy as np

#Api for Vector operation
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

        if operator == 'Add':
            result = [vectorA[0] + vectorB[0], vectorA[1] + vectorB[1]]

        elif operator == 'subtract':
            result = [vectorA[0] - vectorB[0], vectorA[1] - vectorB[1]]

        elif operator == 'multiply':
            result = [scalar*vectorA[0], scalar*vectorB[1]]

        return Response({'result':result})


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
            matrix = matrix
            vector = matrix[0]
            rrf = np.linalg.solve(matrix, vector)
            return Response({'result':rrf})
        
        elif operation == 'determinant':
            determinant = int(np.linalg.det(matrix))
            return Response({'result':determinant})
        
        elif operation == 'rank':
            rank = int(np.linalg.matrix_rank(matrix))
            return Response({'result':rank})
        
        elif operation == 'echelon_form':
            echelon =[np.linalg.inv(matrix)]
            return Response({'result':echelon})
        else:
            return Response({'result':'Invalid Operation Choice'})


#Linear Equation solver
#Used LU Decomposition method.
#need to synchronize variable count according to eqn_LHS and eqn_RHS 
from .serializer import LinearSerializer

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

        result = np.linalg.solve(eqn_LHS,eqn_RHS)
        
        return Response({'result':result})