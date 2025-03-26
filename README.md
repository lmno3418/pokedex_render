# Pokemon Battle Predictor Web App

This is a Flask web application that predicts the winner in a battle between two Pokemon using a pre-trained machine learning model.

## Features

- Advanced Pokemon search functionality:
  - Search by Pokemon ID (e.g., "25" for Pikachu)
  - Search by name (case insensitive)
  - Filter by Pokemon type
  - Filter by generation
- Uses a pre-trained model to predict battle outcomes
- Simple and intuitive web interface
- User authentication system

## How It Works

- **pokemon_db.csv**: Used for populating the Pokemon selection interface
- **pokemon.csv**: Used for the actual ML prediction processing
- **battle_1v1_model.joblib**: Pre-trained ML model that predicts battle outcomes

## Prerequisites

- Python 3.7 or higher
- Required Python packages (see requirements.txt)

## Local Development

1. Clone or download this repository
2. Navigate to the project directory
3. Install required packages:

```
pip install -r requirements.txt
```

4. Create a `.env` file with the following variables:
```
DATABASE_URL=sqlite:///users.db
SECRET_KEY=your-secret-key-here
FLASK_DEBUG=True
```

5. Run the Flask application:

```
python app.py
```

6. Open your web browser and go to http://127.0.0.1:5000/

## Deploying to Render

### Prerequisites
1. Create a [Render](https://render.com) account
2. Create a PostgreSQL database on Render

### Steps to Deploy

1. Create a new Web Service on Render
2. Connect your GitHub repository
3. Configure the following settings:
   - **Name**: Choose a name for your service
   - **Environment**: Python 3
   - **Build Command**: `pip install -r requirements.txt`
   - **Start Command**: `gunicorn app:app`

4. Add the following environment variables:
   - `DATABASE_URL`: Use the Internal Database URL from your Render PostgreSQL database
   - `SECRET_KEY`: A secure random string for session encryption
   - `FLASK_DEBUG`: Set to "False" for production

5. Deploy your application

### Important Notes for Render Deployment
- The app will automatically detect the PostgreSQL database URL
- The first user to register will be able to access the app
- Large files like `battle_1v1_model.joblib` might require extra build time

## How to Use

1. Search for Pokemon using any of these methods:
   - Type a Pokemon ID (e.g., "25")
   - Type a Pokemon name (e.g., "pikachu")
   - Type a Pokemon type (e.g., "fire")
   - Select a type from the dropdown
   - Select a generation from the dropdown
   - Use combinations of the above
   
2. Select your first and second Pokemon for battle from the search results

3. Click "Predict Battle Winner"

4. View the battle prediction result

5. Click "Search for Another Battle" to make another prediction 