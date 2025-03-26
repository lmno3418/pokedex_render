import joblib
import pandas as pd
import json
import os
from flask import (
    Flask,
    render_template,
    request,
    jsonify,
    redirect,
    url_for,
    flash,
    session,
)
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import (
    LoginManager,
    UserMixin,
    login_user,
    login_required,
    logout_user,
    current_user,
)
import re
from dotenv import load_dotenv

# Load environment variables from .env file (if it exists)
load_dotenv()

app = Flask(__name__)
app.config["SECRET_KEY"] = os.environ.get(
    "SECRET_KEY", "pokemon-battle-predictor-secret-key"
)

# Configure the database based on environment variable
database_url = os.environ.get("DATABASE_URL")
# Handle Render PostgreSQL URL format (postgres:// -> postgresql://)
if database_url and database_url.startswith("postgres://"):
    database_url = database_url.replace("postgres://", "postgresql://", 1)

# Set database URI from environment or use SQLite as fallback
app.config["SQLALCHEMY_DATABASE_URI"] = database_url or "sqlite:///users.db"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

# Initialize extensions
db = SQLAlchemy(app)
login_manager = LoginManager(app)
login_manager.login_view = "login"


# User model
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(128), nullable=False)

    def set_password(self, password):
        self.password_hash = generate_password_hash(password)

    def check_password(self, password):
        return check_password_hash(self.password_hash, password)


# Create tables
@app.before_first_request
def create_tables():
    db.create_all()


# User loader
@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


# Load the model
loaded_model = joblib.load("battle_1v1_model.joblib")
print("Loaded model successfully!")

# Load Pokémon data for prediction
pokemon_df = pd.read_csv("pokemon.csv")
pokemon_df["Type 2"] = pokemon_df["Type 2"].fillna("Normal")
pokemon_df["Name"] = pokemon_df["Name"].str.lower()

# Load Pokémon data for selection
pokemon_db_df = pd.read_csv("pokemon_db.csv")
pokemon_db_df["Name"] = pokemon_db_df["Name"].str.lower()


# Prepare pokemon data for frontend search
def prepare_pokemon_data():
    data = []
    for index, row in pokemon_db_df.iterrows():
        pokemon = {
            "id": row["#"],
            "name": row["Name"].title(),
            "type1": row["Type 1"],
            "type2": row["Type 2"] if pd.notna(row["Type 2"]) else None,
            "generation": row["Generation"],
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


# Login route
@app.route("/login", methods=["GET", "POST"])
def login():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        username = request.form.get("username")
        password = request.form.get("password")
        remember = "remember" in request.form

        user = User.query.filter_by(username=username).first()

        if not user or not user.check_password(password):
            flash("Invalid username or password", "danger")
            return redirect(url_for("login"))

        login_user(user, remember=remember)
        next_page = request.args.get("next")

        if not next_page or not next_page.startswith("/"):
            next_page = url_for("index")

        return redirect(next_page)

    return render_template("login.html")


# Register route
@app.route("/register", methods=["GET", "POST"])
def register():
    if current_user.is_authenticated:
        return redirect(url_for("index"))

    if request.method == "POST":
        email = request.form.get("email")
        username = request.form.get("username")
        password = request.form.get("password")
        confirm_password = request.form.get("confirm_password")

        # Validate email
        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        if not re.match(email_pattern, email):
            flash("Invalid email address", "danger")
            return redirect(url_for("register"))

        # Validate username
        if not username or len(username) < 3:
            flash("Username must be at least 3 characters long", "danger")
            return redirect(url_for("register"))

        # Validate password
        if not password or len(password) < 6:
            flash("Password must be at least 6 characters long", "danger")
            return redirect(url_for("register"))

        if password != confirm_password:
            flash("Passwords do not match", "danger")
            return redirect(url_for("register"))

        # Check if email or username already exists
        email_exists = User.query.filter_by(email=email).first()
        username_exists = User.query.filter_by(username=username).first()

        if email_exists:
            flash("Email already registered", "danger")
            return redirect(url_for("register"))

        if username_exists:
            flash("Username already taken", "danger")
            return redirect(url_for("register"))

        # Create new user
        user = User(email=email, username=username)
        user.set_password(password)

        db.session.add(user)
        db.session.commit()

        flash("Registration successful! You can now log in.", "success")
        return redirect(url_for("login"))

    return render_template("register.html")


# Logout route
@app.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("index"))


@app.route("/")
def index():
    # Pass the pokemon data as JSON for the frontend search
    pokemon_data_json = json.dumps(pokemon_frontend_data)
    return render_template("index.html", pokemon_data=pokemon_data_json)


@app.route("/predict", methods=["POST"])
@login_required
def predict():
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
    prediction = loaded_model.predict(input_df)
    result = {
        "winner": "Pokemon 1" if prediction[0] == 1 else "Pokemon 2",
        "pokemon1": p1_name.title(),
        "pokemon2": p2_name.title(),
    }

    return render_template("result.html", result=result)


if __name__ == "__main__":
    debug = os.environ.get("FLASK_DEBUG", "False").lower() == "true"
    app.run(debug=debug)
