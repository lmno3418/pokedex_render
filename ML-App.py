import joblib  # For saving and loading models

# Load the model (if needed later)
loaded_model = joblib.load("battle_1v1_model.joblib")
print("Loaded model successfully!")

# Application.......................................................................................................................
import pandas as pd

# Load Pokémon data
pokemon_df = pd.read_csv("pokemon.csv")
pokemon_df["Type 2"] = pokemon_df["Type 2"].fillna("Normal")

# Normalize Pokémon names to lowercase for case-insensitive matching
pokemon_df["Name"] = pokemon_df["Name"].str.lower()

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
    # Convert input name to lowercase for case-insensitive matching
    name = name.lower()
    if name not in df["Name"].values:
        print(f"No Pokémon exists with the name '{name}'. Please try again.")
        return None

    pokemon = df[df["Name"] == name].iloc[0]
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
        "Legendary": int(pokemon["Legendary"]),
    }


# Get inputs
p1 = input("First Pokémon: ").strip()
p1_data = get_features(
    p1, pokemon_df, TYPE_ENCODINGS["P1_Type1"], TYPE_ENCODINGS["P1_Type2"]
)
if p1_data is None:
    exit()  # Exit if the first Pokémon name is invalid

p2 = input("Second Pokémon: ").strip()
p2_data = get_features(
    p2, pokemon_df, TYPE_ENCODINGS["P2_Type1"], TYPE_ENCODINGS["P2_Type2"]
)
if p2_data is None:
    exit()  # Exit if the second Pokémon name is invalid

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

# Make prediction (assuming model is loaded)
prediction = loaded_model.predict(input_df)
print(f"Prediction: {'P1 Wins' if prediction[0] == 1 else 'P2 Wins'}")
