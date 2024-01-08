from django.shortcuts import render
from joblib import load

model = load('./savedModels/model.joblib')

# Create your views here.

def predictor(request):

    if request.method == 'POST':
        sepal_length = float(request.POST['sepal_length'])
        sepal_width = float(request.POST['sepal_width'])
        petal_length = float(request.POST['petal_length'])
        petal_width = float(request.POST['petal_width'])

        y_pred = model.predict([[sepal_length, sepal_width, petal_length, petal_width]])
        
        if y_pred == 0:
            y_pred = 'Setosa'
        elif y_pred == 1:
            y_pred = 'Versicolor'
        else:
            y_pred = 'Virginica'

        context = {
            'sepal_length': sepal_length,
            'sepal_width': sepal_width,
            'petal_length': petal_length,
            'petal_width': petal_width,
            'y_pred': y_pred
        }

    return render(request, 'main.html', context)