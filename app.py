# fast.py - Shivam's Movie Recommender using FastAPI with Google Drive download

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import joblib
import pandas as pd
import re
import requests
import os
import gdown

app = FastAPI()

# --- Clean title and year extraction for posters ---
def split_title_and_year(title):
    title_str = str(title).strip()
    year = ""
    title_clean = title_str

    year_match = re.search(r'\((\d{4})\)$', title_str)
    if year_match:
        year = year_match.group(1)
        title_clean = title_str[:year_match.start()].strip()

    if title_clean.endswith(', The'):
        title_clean = 'The ' + title_clean[:-5].strip()
    elif title_clean.endswith(', A'):
        title_clean = 'A ' + title_clean[:-3].strip()
    elif title_clean.endswith(', An'):
        title_clean = 'An ' + title_clean[:-4].strip()
    
    title_clean = re.sub(r'\([^)]*\)', '', title_clean).strip()

    return pd.Series([title_clean, year])

# --- CORS Configuration ---
origins = [
    "http://localhost",
    "http://localhost:8000",
    "null",
    "*"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static folder for frontend files
app.mount("/static", StaticFiles(directory="static"), name="static")

# OMDB API Key
OMDB_API_KEY = "3d7c9a51"

# Local folder to store downloaded model files
DATA_DIR = "movie_recommender/static"
os.makedirs(DATA_DIR, exist_ok=True)

# Google Drive file IDs (extracted from your share links)
files_to_download = {
    "knn_model.pkl": "1vWfBWutO0asXe7ebxuZkqJPocVjA-zpB",
    "movie_id_map.pkl": "1rO2CDRhVQ3tnvrk662IeNSBv3SFX8ywG",
    "movies.csv": "1jYcMHmZVGNKs3H5IYRXUZYRakHwE4Z-M",
    "reverse_map.pkl": "15vhK4_E1HfSU-qa53DvpdyOP1hEorpzE",
    "sparse_matrix.pkl": "16cwE0JgT_KSIq-G8fjW6LqKlctfOWIjb",
}

def download_files():
    for filename, file_id in files_to_download.items():
        filepath = os.path.join(DATA_DIR, filename)
        if not os.path.exists(filepath):
            print(f"Downloading {filename} from Google Drive...")
            url = f"https://drive.google.com/uc?id={file_id}"
            try:
                gdown.download(url, filepath, quiet=False)
            except Exception as e:
                print(f"Error downloading {filename}: {e}")
        else:
            print(f"{filename} already exists, skipping download.")

# Download files on startup
download_files()

# Load models and data
knn = None
sparse_matrix = None
movie_id_map = None
reverse_map = None
movies = pd.DataFrame()
all_titles = []

try:
    knn = joblib.load(os.path.join(DATA_DIR, "knn_model.pkl"))
    sparse_matrix = joblib.load(os.path.join(DATA_DIR, "sparse_matrix.pkl"))
    movie_id_map = joblib.load(os.path.join(DATA_DIR, "movie_id_map.pkl"))
    reverse_map = joblib.load(os.path.join(DATA_DIR, "reverse_map.pkl"))
    movies = pd.read_csv(os.path.join(DATA_DIR, "movies.csv"))

    if 'title' in movies.columns and not movies['title'].empty:
        all_titles = movies['title'].dropna().unique().tolist()
        movies[['title_clean', 'year']] = movies['title'].apply(lambda x: split_title_and_year(x))

    print("INFO: All models and data loaded successfully.")
except FileNotFoundError as e:
    print(f"ERROR: Could not load model or data file. Check path: {e}")
except Exception as e:
    print(f"ERROR: Unexpected error loading models/data: {e}")

# Fetch movie details from OMDB API
def fetch_movie_details(title, year=""):
    url = f"http://www.omdbapi.com/?apikey={OMDB_API_KEY}&t={title}&y={year}&plot=short&r=json"
    try:
        res = requests.get(url)
        res.raise_for_status()
        data = res.json()

        if data.get("Response") == "True":
            poster_url = data.get("Poster", "N/A")
            if poster_url == "N/A":
                poster_url = None

            rating = data.get("imdbRating", "N/A")
            plot = data.get("Plot", "N/A")

            return {
                "title": data.get("Title", title),
                "year": data.get("Year", year),
                "poster": poster_url,
                "rating": rating,
                "plot": plot
            }
        else:
            print(f"WARNING: OMDB API did not find details for '{title}' ({year}). Message: {data.get('Error')}")
            return {
                "title": title,
                "year": year,
                "poster": None,
                "rating": "N/A",
                "plot": "No plot available."
            }
    except requests.exceptions.RequestException as e:
        print(f"WARNING: OMDB API request failed for '{title}' ({year}): {e}")
        return {
            "title": title,
            "year": year,
            "poster": None,
            "rating": "N/A",
            "plot": "Error fetching plot."
        }
    except Exception as e:
        print(f"ERROR: Unexpected error in fetch_movie_details for '{title}' ({year}): {e}")
        return {
            "title": title,
            "year": year,
            "poster": None,
            "rating": "N/A",
            "plot": "Error fetching plot."
        }

# Endpoint: Get all movie titles
@app.get("/api/titles")
def get_all_movie_titles():
    if not all_titles:
        print("WARNING: all_titles is empty. Cannot provide titles to frontend.")
        return JSONResponse(content={"error": "Movie titles not loaded."}, status_code=500)
    return JSONResponse(content={"titles": all_titles})

# Endpoint: Get details for a single movie
@app.get("/get-movie-details")
def get_movie_details_by_title(title: str):
    if movies.empty:
        raise HTTPException(status_code=500, detail="Movie data not loaded.")

    movie_row = movies[movies['title'] == title]
    if movie_row.empty:
        raise HTTPException(status_code=404, detail="Movie not found in our database.")

    clean_title = movie_row.iloc[0]['title_clean']
    year = movie_row.iloc[0]['year']

    details = fetch_movie_details(clean_title, year)
    details["original_title"] = title
    return JSONResponse(content=details)

# Serve frontend HTML
@app.get("/")
def serve_index():
    return FileResponse("static/nwpg.html")

# Recommend movies endpoint
@app.get("/recommend")
def recommend_movies(title: str):
    if movies.empty or knn is None or sparse_matrix is None or reverse_map is None or movie_id_map is None:
        print("ERROR: Models or movie data not loaded. Cannot provide recommendations.")
        return JSONResponse(content={"error": "Movie data or models not loaded. Please check server logs."}, status_code=500)

    movie_row = movies[movies['title'] == title]
    if movie_row.empty:
        print(f"DEBUG: Movie '{title}' not found in movies DataFrame.")
        return JSONResponse(content={"error": "Movie not found in our database. Please try a different movie."}, status_code=404)

    movie_id = movie_row.iloc[0]['movieId']
    if movie_id not in reverse_map:
        print(f"DEBUG: Movie ID {movie_id} for '{title}' not in training set (reverse_map).")
        return JSONResponse(content={"error": "Movie not in training set for recommendations. Please try a different movie."}, status_code=404)

    idx = reverse_map[movie_id]

    if idx >= sparse_matrix.shape[0]:
        print(f"ERROR: Index {idx} out of bounds for sparse_matrix with shape {sparse_matrix.shape}.")
        return JSONResponse(content={"error": "Internal error: Model data issue (index out of bounds)."}, status_code=500)

    try:
        distances, indices = knn.kneighbors(sparse_matrix[idx], n_neighbors=6)
    except Exception as e:
        print(f"ERROR: knn.kneighbors failed for index {idx}: {e}")
        return JSONResponse(content={"error": "Error generating recommendations. Model computation issue."}, status_code=500)

    recommendations = []
    for i in indices.flatten()[1:]:
        sim_id = movie_id_map.get(i)
        if sim_id is None:
            print(f"WARNING: No movie ID found for index {i} in movie_id_map.")
            continue

        rec_movie_row = movies[movies['movieId'] == sim_id]
        if rec_movie_row.empty:
            print(f"WARNING: Recommended movie ID {sim_id} not found in original movies DataFrame.")
            continue

        row = rec_movie_row.iloc[0]
        details = fetch_movie_details(row['title_clean'], row['year'])
        details["title"] = row['title']
        recommendations.append(details)

    if not recommendations:
        print(f"DEBUG: No suitable recommendations found for '{title}' after processing.")
        return JSONResponse(content={"error": "Could not find suitable recommendations. Try a different movie."}, status_code=404)

    return JSONResponse(content=recommendations)
