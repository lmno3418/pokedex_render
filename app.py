import joblib
import pandas as pd
import json
import numpy as np
from flask import Flask, render_template, request, jsonify
from dotenv import load_dotenv
import os

# Load environment variables if .env file exists
load_dotenv()

app = Flask(__name__)

# Get file paths from environment variables
MODEL_PATH = os.environ.get("MODEL_PATH", "battle_1v1_model.joblib")
POKEMON_CSV = os.environ.get("POKEMON_CSV", "pokemon.csv")
POKEMON_DB_CSV = os.environ.get("POKEMON_DB_CSV", "pokemon_db.csv")

# Load the model
try:
    loaded_model = joblib.load(MODEL_PATH)
    print(f"Loaded model successfully from {MODEL_PATH}!")
except Exception as e:
    print(f"Error loading model: {e}")
    loaded_model = None

# Load Pokémon data for prediction
pokemon_df = pd.read_csv(POKEMON_CSV)
pokemon_df["Type 2"] = pokemon_df["Type 2"].fillna("Normal")
pokemon_df["Name"] = pokemon_df["Name"].str.lower()

# Load Pokémon data for selection
pokemon_db_df = pd.read_csv(POKEMON_DB_CSV)
pokemon_db_df["Name"] = pokemon_db_df["Name"].str.lower()


# Prepare pokemon data for frontend search
def prepare_pokemon_data():
    data = []
    for index, row in pokemon_db_df.iterrows():
        pokemon = {
            "id": int(row["#"]),
            "name": row["Name"].title(),
            "type1": row["Type 1"],
            "type2": row["Type 2"] if pd.notna(row["Type 2"]) else None,
            "generation": int(row["Generation"]),
            "legendary": bool(row["Legendary"]),
        }
        data.append(pokemon)
    return data


pokemon_frontend_data = prepare_pokemon_data()

# Target encodings from training data
TYPE_ENCODINGS = {
    "P1_Type1": {
        "Bug": 0.4019607843137255,
        "Dark": 0.6054997355896351,
        "Dragon": 0.6137055837563452,
        "Electric": 0.6126840317100792,
        "Fairy": 0.3062200956937799,
        "Fighting": 0.4370860927152318,
        "Fire": 0.5627679118187385,
        "Flying": 0.743801652892562,
        "Ghost": 0.4151133501259446,
        "Grass": 0.4214235377026075,
        "Ground": 0.5137662337662338,
        "Ice": 0.4082721814543029,
        "Normal": 0.5008280887711163,
        "Poison": 0.40166204986149584,
        "Psychic": 0.528969957081545,
        "Rock": 0.3767813694820994,
        "Steel": 0.3742191936399773,
        "Water": 0.4439059158945118,
    },
    "P1_Type2": {
        "Bug": 0.4504950495049505,
        "Dark": 0.5783227848101266,
        "Dragon": 0.5720411663807891,
        "Electric": 0.5142118863049095,
        "Fairy": 0.42644978783592646,
        "Fighting": 0.6700782661047562,
        "Fire": 0.5981554677206851,
        "Flying": 0.6470011439777742,
        "Ghost": 0.19437652811735942,
        "Grass": 0.38664904163912756,
        "Ground": 0.34629133154602326,
        "Ice": 0.5590909090909091,
        "Normal": 0.4395269295751432,
        "Poison": 0.4343675417661098,
        "Psychic": 0.491600790513834,
        "Rock": 0.25792349726775954,
        "Steel": 0.46016381236038717,
        "Water": 0.37219251336898396,
    },
    "P2_Type1": {
        "Bug": 0.540695016003658,
        "Dark": 0.3336745138178096,
        "Dragon": 0.3470948012232416,
        "Electric": 0.3522432332220986,
        "Fairy": 0.6500904159132007,
        "Fighting": 0.5033434650455927,
        "Fire": 0.4023128423615338,
        "Flying": 0.2288135593220339,
        "Ghost": 0.45241809672386896,
        "Grass": 0.5411369633005517,
        "Ground": 0.4404276985743381,
        "Ice": 0.5284810126582279,
        "Normal": 0.42343234323432344,
        "Poison": 0.5424553812871823,
        "Psychic": 0.43596881959910916,
        "Rock": 0.5648280802292264,
        "Steel": 0.5159663865546219,
        "Water": 0.5082550526615428,
    },
    "P2_Type2": {
        "Bug": 0.5522388059701493,
        "Dark": 0.38661417322834646,
        "Dragon": 0.36538461538461536,
        "Electric": 0.45093457943925236,
        "Fairy": 0.5678571428571428,
        "Fighting": 0.31522388059701495,
        "Fire": 0.32249674902470743,
        "Flying": 0.3187540348612008,
        "Ghost": 0.3024390243902439,
        "Grass": 0.55794806839772,
        "Ground": 0.5985853227232537,
        "Ice": 0.3995459704880817,
        "Normal": 0.5069759762238917,
        "Poison": 0.531009738595592,
        "Psychic": 0.46465138956606533,
        "Rock": 0.6940382452193475,
        "Steel": 0.460546282245827,
        "Water": 0.5551801801801802,
    },
}


def get_features(name, df, type1_enc, type2_enc):
    """
    Extract features for a pokemon by name
    """
    # Convert input name to lowercase for case-insensitive matching
    name = name.lower()
    if name not in df["Name"].values:
        return None

    pokemon = df[df["Name"] == name].iloc[0]

    # Handle Legendary which can be either boolean or integer
    legendary_value = pokemon["Legendary"]
    if isinstance(legendary_value, bool):
        legendary = 1 if legendary_value else 0
    else:
        legendary = int(legendary_value)

    return {
        "Type1": type1_enc.get(pokemon["Type 1"], 0),
        "Type2": type2_enc.get(pokemon["Type 2"], 0),
        "HP": pokemon["HP"],
        "Attack": pokemon["Attack"],
        "Defense": pokemon["Defense"],
        "Sp.Atk": pokemon["Sp. Atk"],
        "Sp.Def": pokemon["Sp. Def"],
        "Speed": pokemon["Speed"],
        "Generation": pokemon["Generation"],
        "Legendary": legendary,
    }


@app.route("/")
def index():
    """
    Render the main page with Pokemon selection interface
    """
    # Pass the pokemon data as JSON for the frontend search
    pokemon_data_json = json.dumps(pokemon_frontend_data)
    return render_template("index.html", pokemon_data=pokemon_data_json)


@app.route("/predict", methods=["POST"])
def predict():
    """
    Handle prediction request when form is submitted
    """
    p1_name = request.form.get("pokemon1").lower()
    p2_name = request.form.get("pokemon2").lower()

    # Get Pokemon features from pokemon.csv for ML prediction
    p1_data = get_features(
        p1_name, pokemon_df, TYPE_ENCODINGS["P1_Type1"], TYPE_ENCODINGS["P1_Type2"]
    )
    p2_data = get_features(
        p2_name, pokemon_df, TYPE_ENCODINGS["P2_Type1"], TYPE_ENCODINGS["P2_Type2"]
    )

    if p1_data is None or p2_data is None:
        return jsonify({"error": "One or both Pokemon not found"})

    # Create input DataFrame
    input_data = {
        "P1_Type1": p1_data["Type1"],
        "P1_Type2": p1_data["Type2"],
        "P1_HP": p1_data["HP"],
        "P1_Attack": p1_data["Attack"],
        "P1_Defense": p1_data["Defense"],
        "P1_Sp.Atk": p1_data["Sp.Atk"],
        "P1_Sp.Def": p1_data["Sp.Def"],
        "P1_Speed": p1_data["Speed"],
        "P1_Generation": p1_data["Generation"],
        "P1_Legendary": p1_data["Legendary"],
        "P2_Type1": p2_data["Type1"],
        "P2_Type2": p2_data["Type2"],
        "P2_HP": p2_data["HP"],
        "P2_Attack": p2_data["Attack"],
        "P2_Defense": p2_data["Defense"],
        "P2_Sp.Atk": p2_data["Sp.Atk"],
        "P2_Sp.Def": p2_data["Sp.Def"],
        "P2_Speed": p2_data["Speed"],
        "P2_Generation": p2_data["Generation"],
        "P2_Legendary": p2_data["Legendary"],
    }

    input_df = pd.DataFrame([input_data])

    # Make prediction
    try:
        prediction = loaded_model.predict(input_df)
        result = {
            "winner": "Pokemon 1" if prediction[0] == 1 else "Pokemon 2",
            "pokemon1": p1_name.title(),
            "pokemon2": p2_name.title(),
        }
        return render_template("result.html", result=result)
    except Exception as e:
        return jsonify({"error": f"Prediction error: {str(e)}"})


# Health check endpoint for Render
@app.route("/health")
def health_check():
    return jsonify({"status": "healthy", "model_loaded": loaded_model is not None})


if __name__ == "__main__":
    # Get port from environment variable (Render will set this)
    port = int(os.environ.get("PORT", 5000))

    # In production, debug should be False
    debug_mode = os.environ.get("FLASK_DEBUG", "False").lower() == "true"

    print(
        f"Starting server on port {port} with debug mode {'enabled' if debug_mode else 'disabled'}"
    )
    app.run(host="0.0.0.0", port=port, debug=debug_mode)
