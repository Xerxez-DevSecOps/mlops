from django.shortcuts import render
from rest_framework.decorators import api_view
from rest_framework.response import Response  # <-- Typo in your original code
import numpy as np
import joblib
import os
from .serializers import InsuranceSerializer

# Correct model path using absolute path
model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'models', 'model.joblib')
model_path = os.path.normpath(model_path)  # Normalize path to handle '..'

# Load the model
model = joblib.load(model_path)

@api_view(['POST'])
def predict(request):
    if request.method == 'POST':
        # Deserialize the input data from the request
        serializer = InsuranceSerializer(data=request.data)
        if serializer.is_valid():
            input_data = tuple(serializer.validated_data.values())
            input_data_as_numpy_array = np.asarray(input_data)
            input_data_reshaped = input_data_as_numpy_array.reshape(1, -1)
            print(input_data_reshaped)

            # Make a prediction using the model
            prediction = model.predict(input_data_reshaped)

            # Return the prediction as a JSON response
            return Response({'prediction': prediction.tolist()})
        else:
            return Response(serializer.errors, status=400)
