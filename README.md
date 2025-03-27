# Pokemon Battle Predictor Web App

This is a Flask web application that predicts the winner in a battle between two Pokemon using a pre-trained machine learning model.

## Features

- Advanced Pokemon search functionality:
  - Search by Pokemon ID (e.g., "25" for Pikachu)
  - Search by name (case insensitive)
  - Filter by Pokemon type 1 and type 2 separately
  - Filter by generation
- Uses a pre-trained model to predict battle outcomes
- Simple and intuitive web interface

## How It Works

- **pokemon_db.csv**: Used for populating the Pokemon selection interface
- **pokemon.csv**: Used for the actual ML prediction processing
- **battle_1v1_model.joblib**: Pre-trained ML model that predicts battle outcomes

## Prerequisites

- Python 3.10 or higher
- Required Python packages (see requirements.txt)

## Installation

1. Clone or download this repository
2. Navigate to the project directory
3. Install required packages:

```
pip install -r requirements.txt
```

## Configuration

The application uses environment variables for configuration. You can set these in a `.env` file:

```
# Flask application configuration
FLASK_APP=app.py
FLASK_DEBUG=True

# Server configuration
PORT=5000

# Application settings
MODEL_PATH=battle_1v1_model.joblib
POKEMON_CSV=pokemon.csv
POKEMON_DB_CSV=pokemon_db.csv
```

## Running Locally

1. Make sure you have the following files in the project directory:
   - pokemon.csv
   - pokemon_db.csv
   - battle_1v1_model.joblib

2. Run the Flask application:

```
python app.py
```

Or using Flask CLI:

```
flask run
```

3. Open your web browser and go to http://127.0.0.1:5000/

## Deployment on Render

This application is configured for easy deployment on Render. Follow these steps:

1. Create a new account on [Render](https://render.com/) if you don't have one.

2. Create a new Web Service on Render and connect to your GitHub repository.

3. Use the following settings:
   - **Name**: pokemon-battle-predictor (or your preferred name)
   - **Environment**: Python
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`

4. Add the following environment variables in the Render dashboard:
   - `FLASK_APP`: app.py
   - `FLASK_DEBUG`: false
   - `PYTHON_VERSION`: 3.10.0

5. Set the Health Check Path to `/health`

6. Upload your data files to the service:
   - pokemon.csv
   - pokemon_db.csv
   - battle_1v1_model.joblib

7. Deploy your service!

### Automatic Deployment with render.yaml

If you prefer, you can use the included `render.yaml` file for automatic configuration:

1. Fork this repository
2. Create a new Blueprint on Render and connect to your GitHub repository
3. Render will automatically configure the service based on the settings in `render.yaml`

## How to Use

1. Search for Pokemon using any of these methods:
   - Type a Pokemon ID (e.g., "25")
   - Type a Pokemon name (e.g., "pikachu")
   - Type a Pokemon type (e.g., "fire")
   - Select a primary type from the "Type 1" dropdown
   - Select a secondary type from the "Type 2" dropdown
   - Select a generation from the dropdown
   - Use combinations of the above
   
2. Select your first and second Pokemon for battle from the search results

3. Click "Predict Battle Winner"

4. View the battle prediction result

5. Click "Search for Another Battle" to make another prediction

## Dependencies

The application requires specific versions of libraries which are listed in the requirements.txt file. Key dependencies include:

- Flask 3.1.0
- pandas 2.2.3
- scikit-learn 1.6.1
- joblib 1.4.2
- numpy 2.2.4
- gunicorn 21.2.0 (for production deployment) 