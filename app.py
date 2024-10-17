import numpy as np

def predict_win(features_dict):
    # Define the feature order and ensure all features are included
    feature_names = [
        'DBNOs', 'headshotKills', 'killPlace', 'killPoints', 'killStreaks',
        'longestKill', 'numGroups', 'rankPoints', 'roadKills', 'teamKills',
        'vehicleDestroys', 'weaponsAcquired', 'winPoints', 'playerJoined',
        'totalDistance', 'headshot_rate', 'killsNorm', 'damageDealtNorm',
        'maxPlaceNorm', 'matchDurationNorm', 'traveldistance', 'healsnboosts',
        'assist', 'matchType_crashfpp', 'matchType_crashtpp', 'matchType_duo',
        'matchType_duo-fpp', 'matchType_flarefpp', 'matchType_flaretpp',
        'matchType_normal-duo', 'matchType_normal-duo-fpp',
        'matchType_normal-solo', 'matchType_normal-solo-fpp',
        'matchType_normal-squad', 'matchType_normal-squad-fpp',
        'matchType_solo', 'matchType_solo-fpp', 'matchType_squad',
        'matchType_squad-fpp', 'killswithoutMoving_False'
    ]
    
    # Create feature vector with the correct order
    feature_vector = [features_dict.get(name, 0) for name in feature_names]
    
    # Convert to numpy array and reshape for prediction
    feature_array = np.array([feature_vector])
    
    # Predict using the model
    prediction = model.predict(feature_array)
    return prediction

import streamlit as st
import pickle

# Load the model
with open("catboost_model.pkl", "rb") as file:
    model = pickle.load(file)

# Streamlit app code
def main():
    st.title("PUBG Win Prediction")

    # Input fields for the user to enter data
    DBNOs = st.number_input("DBNOs", min_value=0)
    headshotKills = st.number_input("Headshot Kills", min_value=0)
    killPlace = st.number_input("Kill Place", min_value=0)
    killPoints = st.number_input("Kill Points", min_value=0)
    killStreaks = st.number_input("Kill Streaks", min_value=0)
    longestKill = st.number_input("Longest Kill", min_value=0)
    numGroups = st.number_input("Number of Groups", min_value=0)
    rankPoints = st.number_input("Rank Points", min_value=0)
    roadKills = st.number_input("Road Kills", min_value=0)
    teamKills = st.number_input("Team Kills", min_value=0)
    vehicleDestroys = st.number_input("Vehicle Destroys", min_value=0)
    weaponsAcquired = st.number_input("Weapons Acquired", min_value=0)
    winPoints = st.number_input("Win Points", min_value=0)
    playerJoined = st.number_input("Player Joined", min_value=0)
    totalDistance = st.number_input("Total Distance", min_value=0)
    headshot_rate = st.number_input("Headshot Rate", min_value=0.0)
    killsNorm = st.number_input("Kills Norm", min_value=0.0)
    damageDealtNorm = st.number_input("Damage Dealt Norm", min_value=0.0)
    maxPlaceNorm = st.number_input("Max Place Norm", min_value=0.0)
    matchDurationNorm = st.number_input("Match Duration Norm", min_value=0.0)
    traveldistance = st.number_input("Travel Distance", min_value=0.0)
    healsnboosts = st.number_input("Heals and Boosts", min_value=0)
    assist = st.number_input("Assist", min_value=0)
    
    # Match type inputs (if applicable)
    matchType = st.selectbox("Match Type", [
        'matchType_crashfpp', 'matchType_crashtpp', 'matchType_duo',
        'matchType_duo-fpp', 'matchType_flarefpp', 'matchType_flaretpp',
        'matchType_normal-duo', 'matchType_normal-duo-fpp',
        'matchType_normal-solo', 'matchType_normal-solo-fpp',
        'matchType_normal-squad', 'matchType_normal-squad-fpp',
        'matchType_solo', 'matchType_solo-fpp', 'matchType_squad',
        'matchType_squad-fpp'
    ])

    # Create the feature dictionary
    features_dict = {
        'DBNOs': DBNOs,
        'headshotKills': headshotKills,
        'killPlace': killPlace,
        'killPoints': killPoints,
        'killStreaks': killStreaks,
        'longestKill': longestKill,
        'numGroups': numGroups,
        'rankPoints': rankPoints,
        'roadKills': roadKills,
        'teamKills': teamKills,
        'vehicleDestroys': vehicleDestroys,
        'weaponsAcquired': weaponsAcquired,
        'winPoints': winPoints,
        'playerJoined': playerJoined,
        'totalDistance': totalDistance,
        'headshot_rate': headshot_rate,
        'killsNorm': killsNorm,
        'damageDealtNorm': damageDealtNorm,
        'maxPlaceNorm': maxPlaceNorm,
        'matchDurationNorm': matchDurationNorm,
        'traveldistance': traveldistance,
        'healsnboosts': healsnboosts,
        'assist': assist,
        'matchType': matchType
    }

    # Predict and display the result
    prediction = predict_win(features_dict)
    st.write(f"Predicted probability of getting chicken dinner: {prediction[0]:.2f}")

if __name__ == "__main__":
    main()
