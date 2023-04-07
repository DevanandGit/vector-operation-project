from rest_framework import serializers

class VectorOperationSerializer(serializers.Serializer):
    vectorA = serializers.CharField(max_length=5)
    vectorB = serializers.CharField(max_length=5, default = "0,0")
    scalar = serializers.IntegerField(default=1)
    operator = serializers.ChoiceField(choices=['Add', 'subtract', 'multiply'])