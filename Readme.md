# Hybrid AI Movie Recommendation System

An intelligent AI-based Movie Recommendation System that combines classical AI techniques (search algorithms, CSP, heuristics) with Machine Learning (K-Means + Neural Networks) and provides results through an interactive Streamlit dashboard.

---

## Features

- Constraint Satisfaction Problem (CSP) filtering  
- Search Algorithms: BFS, DFS, A*  
- Heuristic-based ranking system  
- K-Means clustering for grouping movies/users  
- ANN (Neural Network) for rating prediction (PyTorch / TensorFlow)  
- Interactive Streamlit dashboard  
- Explainable recommendations (reason for each suggestion)

---

## Tech Stack

Python, Streamlit, Pandas, NumPy, Scikit-learn, PyTorch, Plotly

---

## Project Structure

```
Movie Recommendation System/
├── app.py
├── requirements.txt
├── data/
│ └── imdb_dataset.csv
├── modules/
│ ├── csp.py
│ ├── search.py
│ ├── heuristic.py
│ └── ml_model.py
├── utils/
│ └── helpers.py
└── models/
└── trained_model.pth
```
---

## Dataset

Uses IMDb Top Movies dataset containing:
- Movie Title  
- Genre  
- Year  
- Rating  
- Runtime  
- Meta Score  

Place dataset inside the `data/` folder.

---

## Installation

git clone https://github.com/mirza1272/Movie-Recommendation-System.git  
cd Movie-Recommendation-System  

python -m venv venv  
venv\Scripts\activate  

pip install -r requirements.txt  

---

## Run Project

streamlit run app.py  

---

## Working Flow

1. User enters preferences (genre, year, rating, duration)  
2. CSP filters valid movies  
3. BFS / DFS / A* explores search space  
4. Heuristic ranks results  
5. K-Means clusters similar movies/users  
6. ANN predicts ratings  
7. Final recommendations are shown in UI  

---

## Output

- Top recommended movies  
- Predicted ratings  
- Match score  
- Explanation for recommendation  

---

## Future Improvements

- Add collaborative filtering  
- Improve ANN accuracy  
- Deploy on Streamlit Cloud  
- Integrate real IMDb API  

---


## Author

**Haseeb ur Rahman**  

### Collaborators

- Areeba Majeed
- Maheen Fatima 

---

## License

For educational purposes only