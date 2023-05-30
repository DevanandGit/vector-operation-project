from rest_framework import serializers

#problem-1
class VectorOperationSerializer(serializers.Serializer):
    vectorA = serializers.CharField(max_length=5)
    vectorB = serializers.CharField(max_length=5, default = "0,0")
    scalar = serializers.IntegerField(default=1)
    operator = serializers.ChoiceField(choices=['add', 'subtract', 'multiply'])

#problem-2
class MatrixOperationSerializer(serializers.Serializer):
    row = serializers.IntegerField()
    col = serializers.IntegerField()
    matrix = serializers.ListField(child=serializers.ListField(child=serializers.IntegerField()))
    operation = serializers.ChoiceField(choices=['row_reduced_form', 'determinant', 'rank']) #'echelon_form'])

    # def __init__(self, *args, **kwargs):
    #     self.matrix = serializers.ListField(child=serializers.ListField(child=serializers.IntegerField(),max_length = self.col,max_length = self.col), max_length = self.row,max_length = self.row)
    #     return super.__init__(*args, **kwargs)

#problem-3
class LinearSerializer(serializers.Serializer):
    variable_count = serializers.IntegerField()
    eqn_LHS = serializers.ListField(child=serializers.ListField(child=serializers.IntegerField()))
    eqn_RHS = serializers.ListField(child=serializers.IntegerField())

#problem-4
# class LinearTransformationSerializer(serializers.Serializer):
#     image = serializers.ImageField()
#     matrix = serializers.ListField(child=serializers.ListField(child=serializers.IntegerField()))

#problem - 5
class EigenValueSpace(serializers.Serializer):
    matrix = serializers.ListField(child=serializers.ListField(child=serializers.IntegerField()))

#problem - 6
class MatrixDecomposition(serializers.Serializer):
    matrix = serializers.ListField(child=serializers.ListField(child=serializers.IntegerField()))

#problem - 7
class LinearRegression(serializers.Serializer):
    data1 = serializers.ListField(child = serializers.IntegerField())
    data2 = serializers.ListField(child = serializers.IntegerField())

#problem -8
class PCASerializer(serializers.Serializer):
    matrix = serializers.ListField(child=serializers.ListField(child=serializers.IntegerField()))

#problem - 9
class SVD(serializers.Serializer):
    image = serializers.ImageField()
    k = serializers.IntegerField()

