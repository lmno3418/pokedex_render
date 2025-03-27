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

## Running the Application

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