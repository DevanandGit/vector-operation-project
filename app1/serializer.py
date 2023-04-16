from rest_framework import serializers

class VectorOperationSerializer(serializers.Serializer):
    vectorA = serializers.CharField(max_length=5)
    vectorB = serializers.CharField(max_length=5, default = "0,0")
    scalar = serializers.IntegerField(default=1)
    operator = serializers.ChoiceField(choices=['Add', 'subtract', 'multiply'])

class MatrixOperationSerializer(serializers.Serializer):
    row = serializers.IntegerField()
    col = serializers.IntegerField()
    matrix = serializers.ListField(child=serializers.ListField(child=serializers.IntegerField()))
    operation = serializers.ChoiceField(choices=['row_reduced_form', 'determinant', 'rank', 'echelon_form'])

class LinearSerializer(serializers.Serializer):
    variable_count = serializers.IntegerField()
    eqn_LHS = serializers.ListField(child=serializers.ListField(child=serializers.IntegerField()))
    eqn_RHS = serializers.ListField(child=serializers.IntegerField())