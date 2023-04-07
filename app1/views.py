from django.shortcuts import render
from rest_framework.views import APIView
from .serializer import VectorOperationSerializer
from rest_framework.response import Response
from rest_framework import generics

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
                # return result
        elif operator == 'subtract':
            result = [vectorA[0] - vectorB[0], vectorA[1] - vectorB[1]]
                # return result
        elif operator == 'multiply':
            result = [scalar*vectorA[0], scalar*vectorB[1]]
                # return result
        return Response({'result':result})
        
        

      
      

