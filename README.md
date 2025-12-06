# 🎮 PUBG Game Prediction 🏆

![PUBG Jump from Plane](https://w0.peakpx.com/wallpaper/505/66/HD-wallpaper-pubg-jump-from-plane-pubg-playerunknowns-battlegrounds-2018-games-games-thumbnail.jpg)

## 🚀 Overview
This project aims to **predict a player's probability of getting a chicken dinner** in the popular game **PUBG** using various in-game statistics. By analyzing key features such as kills, assists, and damage dealt, the model estimates the likelihood of a player winning a match. This can help players gauge their expected performance and refine their strategies.
## ✨ Features
- 🧠 **Predict player win percentages** based on in-game statistics.
- 📊 **Analyze critical data points** such as kills, assists, and damage dealt.
- 🤖 **Compare the performance** of different machine learning models to find the most accurate one.
- 🔍 **Feature importance insights** showing the impact on win percentage prediction.

## ⚙️ Models Used
This project uses the following machine learning models:
- **CatBoost Model**: A gradient boosting algorithm known for its high performance and accuracy.
- **Linear Regression**: A basic regression technique used for comparison.

## 📈 Model Performance
| Model                | RMSE  | R²    |
|----------------------|-------|-------|
| **CatBoost Model**   | 0.08  | 0.93  |
| **Linear Regression** | 0.14  | 0.81  |

## 📂 Dataset
The dataset includes a wide range of features such as:
- `assists`
- `kills`
- `damageDealt`
- `matchType`
- `winPlacePerc` (the target variable for win percentage)
- **etc.**

For details about the dataset, please refer to the [dataset_info.md](dataset_info.md) file.

These features are crucial for understanding player behavior and predicting their potential ranking in a match.

## 🛠️ How to Run the Project
1. Clone the repository:
    ```bash
    git clone https://github.com/NeerajAnthal/PUBG-GAME-PREDICTION.git
    ```
2. Install the necessary dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3. Run the notebook to train the model:
    ```bash
    jupyter notebook PUBG_Prediction.ipynb
    ```

## 💻 Installation
Make sure you have Python 3.x installed. Install dependencies using:
```bash
pip install -r requirements.txt
