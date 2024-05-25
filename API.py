# Importación librerías
from flask import Flask
from flask_restx import Api, Resource, fields
from flask_cors import CORS
import joblib
import pandas as pd

# Definición de la aplicación Flask
app = Flask(__name__)
CORS(app)

# Definición de la API Flask
api = Api(
    app, 
    version='1.0', 
    title='Predicción de Géneros Cinematográficos',
    description='El objetivo es predecir la probabilidad de que una película pertenezca a uno o varios de los siguientes géneros: Acción, Aventura, Animación, Biografía, Comedia, Crimen, Documental, Drama, Familia, Fantasía, Film-Noir, Historia, Terror, Música, Musical, Misterio, Noticias, Romance, Ciencia Ficción, Cortometraje, Deporte, Suspenso, Guerra o Western'
)

# Definición de argumentos o parámetros de la API
parser = api.parser()
parser.add_argument('Titulo', type=str, required=True, help='Titulo o nombre de la pelicula', location='args')
parser.add_argument('Sinopsis', type=str, required=True, help='Sinopsis de la pelicula', location='args')

# Cargar el modelo entrenado
pipeline = joblib.load('modelo_prediccion_genero_pelicula.pkl')

# Lista de géneros
genres = ['Action', 'Adventure', 'Animation', 'Biography', 'Comedy', 'Crime', 'Documentary', 'Drama', 'Family',
          'Fantasy', 'Film-Noir', 'History', 'Horror', 'Music', 'Musical', 'Mystery', 'News', 'Romance',
          'Sci-Fi', 'Short', 'Sport', 'Thriller', 'War', 'Western']

# Definición de campos de recurso
resource_fields = api.model('Resource', {
    'result': fields.List(fields.String),
    'probabilities': fields.List(fields.String)
})

class PrediccionGeneroApi(Resource):
    @api.expect(parser)
    @api.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()
        title = args['Titulo']
        plot = args['Sinopsis']
        
        # Concatenar título y sinopsis
        X = title + " " + plot
        
        # Realizar la predicción
        y_pred_prob = pipeline.predict_proba([X])[0]
        
        # Convertir predicciones a nombres de géneros y probabilidades
        results = {genres[i]: prob for i, prob in enumerate(y_pred_prob)}
        
        # Filtrar géneros con probabilidad mayor a 0.5
        filtered_genres = [genre for genre, prob in results.items() if prob > 0.5]
        
        # Crear lista de probabilidades en formato requerido
        predicted_genres = [f"{genre}: {prob:.6f}" for genre, prob in results.items()]
        
        return {'result': filtered_genres, 'probabilities': predicted_genres}, 200

# Añadir recurso a la API
api.add_resource(PrediccionGeneroApi, '/genero_pelicula') 

# Ejecución de la aplicación que disponibiliza el modelo de manera local en el puerto 5000
if __name__ == '__main__':
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)