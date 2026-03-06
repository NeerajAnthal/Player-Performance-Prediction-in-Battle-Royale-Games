import numpy as np
import streamlit as st
import pickle

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PUBG Win Predictor",
    page_icon="🐔",
    layout="wide",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Rajdhani:wght@400;600;700&family=Share+Tech+Mono&display=swap');

html, body, [class*="css"] {
    font-family: 'Rajdhani', sans-serif;
    background-color: #0d0f14;
    color: #e0e4ef;
}

.main { background-color: #0d0f14; }

h1, h2, h3 { font-family: 'Rajdhani', sans-serif; font-weight: 700; }

.stNumberInput > label, .stSelectbox > label, .stSlider > label {
    font-family: 'Share Tech Mono', monospace;
    font-size: 0.78rem;
    color: #7a8baa;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

.block-container { padding: 2rem 3rem; }

.section-header {
    font-size: 0.72rem;
    font-family: 'Share Tech Mono', monospace;
    letter-spacing: 0.2em;
    color: #f0c040;
    text-transform: uppercase;
    border-bottom: 1px solid #2a3045;
    padding-bottom: 6px;
    margin-bottom: 1rem;
    margin-top: 1.5rem;
}

.predict-box {
    background: linear-gradient(135deg, #1a1f2e 0%, #111520 100%);
    border: 1px solid #2a3045;
    border-radius: 12px;
    padding: 2rem;
    text-align: center;
    margin-top: 2rem;
}

.result-value {
    font-family: 'Share Tech Mono', monospace;
    font-size: 4rem;
    font-weight: 700;
    color: #f0c040;
    line-height: 1;
}

.result-label {
    font-size: 0.85rem;
    color: #7a8baa;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.4rem;
}

.chicken-banner {
    font-size: 1.1rem;
    color: #4ade80;
    font-family: 'Share Tech Mono', monospace;
    margin-top: 0.8rem;
}

.danger-banner {
    font-size: 1.1rem;
    color: #f87171;
    font-family: 'Share Tech Mono', monospace;
    margin-top: 0.8rem;
}

div[data-testid="stNumberInput"] input {
    background-color: #161b27;
    border: 1px solid #2a3045;
    color: #e0e4ef;
    font-family: 'Share Tech Mono', monospace;
    border-radius: 6px;
}

div[data-testid="stSelectbox"] > div > div {
    background-color: #161b27;
    border: 1px solid #2a3045;
    color: #e0e4ef;
    border-radius: 6px;
}

.stButton > button {
    width: 100%;
    background: linear-gradient(135deg, #f0c040 0%, #d4a820 100%);
    color: #0d0f14;
    font-family: 'Rajdhani', sans-serif;
    font-weight: 700;
    font-size: 1.1rem;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border: none;
    border-radius: 8px;
    padding: 0.75rem;
    margin-top: 1.5rem;
    cursor: pointer;
    transition: opacity 0.2s;
}
.stButton > button:hover { opacity: 0.88; }
</style>
""", unsafe_allow_html=True)

# ── Load model ─────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    with open("catboost_model.pkl", "rb") as f:
        return pickle.load(f)

try:
    model = load_model()
    model_loaded = True
except FileNotFoundError:
    model_loaded = False

# ── Feature builder ─────────────────────────────────────────────────────────────
MATCH_TYPES = [
    'crashfpp', 'crashtpp', 'duo', 'duo-fpp', 'flarefpp', 'flaretpp',
    'normal-duo', 'normal-duo-fpp', 'normal-solo', 'normal-solo-fpp',
    'normal-squad', 'normal-squad-fpp', 'solo', 'solo-fpp', 'squad', 'squad-fpp'
]

FEATURE_ORDER = [
    'DBNOs', 'headshotKills', 'killPlace', 'killPoints', 'killStreaks',
    'longestKill', 'numGroups', 'rankPoints', 'roadKills', 'teamKills',
    'vehicleDestroys', 'weaponsAcquired', 'winPoints', 'playerJoined',
    'totalDistance', 'headshot_rate', 'killsNorm', 'damageDealtNorm',
    'maxPlaceNorm', 'matchDurationNorm', 'traveldistance', 'healsnboosts',
    'assist',
    # one-hot: matchType
    'matchType_crashfpp', 'matchType_crashtpp', 'matchType_duo',
    'matchType_duo-fpp', 'matchType_flarefpp', 'matchType_flaretpp',
    'matchType_normal-duo', 'matchType_normal-duo-fpp',
    'matchType_normal-solo', 'matchType_normal-solo-fpp',
    'matchType_normal-squad', 'matchType_normal-squad-fpp',
    'matchType_solo', 'matchType_solo-fpp', 'matchType_squad',
    'matchType_squad-fpp',
    # one-hot: killswithoutMoving (drop_first=True keeps only _False or _True)
    'killswithoutMoving_False',
]

def build_feature_vector(inputs: dict, match_type: str, kills_without_moving: bool) -> np.ndarray:
    """
    Convert raw inputs into the same feature vector the model was trained on.
    Handles one-hot encoding for matchType and killswithoutMoving internally.
    """
    vec = {name: 0 for name in FEATURE_ORDER}

    # Scalar features
    for k, v in inputs.items():
        if k in vec:
            vec[k] = v

    # One-hot: matchType  →  matchType_<value>
    col = f"matchType_{match_type}"
    if col in vec:
        vec[col] = 1

    # One-hot: killswithoutMoving (get_dummies drop_first=True keeps _False)
    # If kills_without_moving is False  → killswithoutMoving_False = 1
    # If kills_without_moving is True   → killswithoutMoving_False = 0
    vec['killswithoutMoving_False'] = int(not kills_without_moving)

    return np.array([[vec[f] for f in FEATURE_ORDER]])

# ── UI ─────────────────────────────────────────────────────────────────────────
st.markdown("# 🐔 PUBG WIN PREDICTOR")
st.markdown("<p style='color:#7a8baa;font-size:0.9rem;margin-top:-0.5rem;'>Feed your match stats — find out if you're getting that Chicken Dinner.</p>", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ `catboost_model.pkl` not found. Place it in the same directory as this script.")
    st.stop()

# ── Section 1: Combat ──────────────────────────────────────────────────────────
st.markdown('<div class="section-header">⚔️ Combat Stats</div>', unsafe_allow_html=True)
c1, c2, c3, c4 = st.columns(4)
DBNOs           = c1.number_input("DBNOs",            min_value=0, value=0)
headshotKills   = c2.number_input("Headshot Kills",   min_value=0, value=0)
killStreaks      = c3.number_input("Kill Streaks",     min_value=0, value=0)
longestKill     = c4.number_input("Longest Kill (m)", min_value=0, value=0)

c5, c6, c7, c8 = st.columns(4)
roadKills       = c5.number_input("Road Kills",       min_value=0, value=0)
teamKills       = c6.number_input("Team Kills",       min_value=0, value=0)
vehicleDestroys = c7.number_input("Vehicle Destroys", min_value=0, value=0)
assist          = c8.number_input("Assists",          min_value=0, value=0)

kills_without_moving = st.checkbox("Did player get kills WITHOUT moving?", value=False)

# ── Section 2: Points & Rank ───────────────────────────────────────────────────
st.markdown('<div class="section-header">🏆 Points & Ranking</div>', unsafe_allow_html=True)
p1, p2, p3, p4 = st.columns(4)
killPlace   = p1.number_input("Kill Place",   min_value=0, value=0)
killPoints  = p2.number_input("Kill Points",  min_value=0, value=0)
rankPoints  = p3.number_input("Rank Points",  min_value=0, value=0)
winPoints   = p4.number_input("Win Points",   min_value=0, value=0)

# ── Section 3: Movement & Loot ─────────────────────────────────────────────────
st.markdown('<div class="section-header">🎒 Movement & Loot</div>', unsafe_allow_html=True)
m1, m2, m3, m4 = st.columns(4)
totalDistance   = m1.number_input("Total Distance",    min_value=0,   value=0)
traveldistance  = m2.number_input("Travel Distance",   min_value=0.0, value=0.0)
weaponsAcquired = m3.number_input("Weapons Acquired",  min_value=0,   value=0)
healsnboosts    = m4.number_input("Heals & Boosts",    min_value=0,   value=0)

# ── Section 4: Normalised / Derived ───────────────────────────────────────────
st.markdown('<div class="section-header">📊 Normalised & Derived Features</div>', unsafe_allow_html=True)
n1, n2, n3 = st.columns(3)
headshot_rate    = n1.number_input("Headshot Rate",        min_value=0.0, max_value=1.0, value=0.0, format="%.3f")
killsNorm        = n2.number_input("Kills Norm",           min_value=0.0, value=0.0, format="%.4f")
damageDealtNorm  = n3.number_input("Damage Dealt Norm",    min_value=0.0, value=0.0, format="%.4f")

n4, n5 = st.columns(2)
maxPlaceNorm      = n4.number_input("Max Place Norm",      min_value=0.0, value=0.0, format="%.4f")
matchDurationNorm = n5.number_input("Match Duration Norm", min_value=0.0, value=0.0, format="%.4f")

# ── Section 5: Match Info ──────────────────────────────────────────────────────
st.markdown('<div class="section-header">🗺️ Match Info</div>', unsafe_allow_html=True)
mi1, mi2, mi3 = st.columns(3)
numGroups    = mi1.number_input("Num Groups",    min_value=0, value=0)
playerJoined = mi2.number_input("Player Joined", min_value=0, value=0)
matchType    = mi3.selectbox("Match Type", MATCH_TYPES)

# ── Predict ────────────────────────────────────────────────────────────────────
if st.button("🐔 PREDICT CHICKEN DINNER"):
    raw_inputs = {
        'DBNOs': DBNOs, 'headshotKills': headshotKills, 'killPlace': killPlace,
        'killPoints': killPoints, 'killStreaks': killStreaks, 'longestKill': longestKill,
        'numGroups': numGroups, 'rankPoints': rankPoints, 'roadKills': roadKills,
        'teamKills': teamKills, 'vehicleDestroys': vehicleDestroys,
        'weaponsAcquired': weaponsAcquired, 'winPoints': winPoints,
        'playerJoined': playerJoined, 'totalDistance': totalDistance,
        'headshot_rate': headshot_rate, 'killsNorm': killsNorm,
        'damageDealtNorm': damageDealtNorm, 'maxPlaceNorm': maxPlaceNorm,
        'matchDurationNorm': matchDurationNorm, 'traveldistance': traveldistance,
        'healsnboosts': healsnboosts, 'assist': assist,
    }

    feature_array = build_feature_vector(raw_inputs, matchType, kills_without_moving)
    prediction = model.predict(feature_array)[0]
    pct = float(prediction) * 100

    st.markdown(f"""
    <div class="predict-box">
        <div class="result-value">{pct:.1f}%</div>
        <div class="result-label">Predicted Win Placement Percentile</div>
        <div class="{'chicken-banner' if pct >= 70 else 'danger-banner'}">
            {'🏆 CHICKEN DINNER IS YOURS!' if pct >= 85 else '✅ Strong performance — top contender.' if pct >= 70 else '⚠️ Mid-game grind needed.' if pct >= 40 else '💀 Rough match — keep practising.'}
        </div>
    </div>
    """, unsafe_allow_html=True)

    with st.expander("🔬 Raw feature vector (debug)"):
        import pandas as pd
        debug_df = pd.DataFrame(feature_array, columns=FEATURE_ORDER).T
        debug_df.columns = ["value"]
        st.dataframe(debug_df, use_container_width=True)
