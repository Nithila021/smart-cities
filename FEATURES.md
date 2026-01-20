# Smart Cities Crime Safety Analysis System – Features

This file summarizes all major features of the project, grouped by what the end user experiences and what happens behind the scenes.

---

## 1. User-Facing Features

- **Real-time safety analysis for NYC locations**  
  - User can type a natural-language query like `"Restaurant, Times Square, New York"` and receive a safety assessment.
- **Interactive chat interface**  
  - Chat-style UI for asking questions about locations.  
  - Shows message history and safety responses in conversation form.
- **Location-based safety score (0–100)**  
  - Overall safety rating for the queried area.  
  - Includes supporting context such as nearby crime count and dominant crime types.
- **Time-of-day and context-aware information**  
  - Analysis includes time-of-day patterns and crime density classification for the queried area.
- **Amenity-aware responses**  
  - Queries follow an `[Amenity], [Address]` format (e.g., restaurant, park, school) and responses integrate amenity context.

---

## 2. Mapping & Visualization Features

- **Interactive map (Leaflet)**  
  - Visualizes crime clusters and zones directly on a map of New York City.  
  - Supports pan/zoom and interactive viewing.
- **DBSCAN crime cluster visualization**  
  - Shows crime hotspots as map markers (e.g., pink markers).
- **Demographic zone visualization**  
  - Displays zones derived from demographic clustering (e.g., dark pink markers).
- **Crime density visualization**  
  - Shows low/medium/high crime density zones (e.g., burgundy markers) derived from KDE.
- **Multi-layer map controls**  
  - Users can toggle overlays for:  
    - DBSCAN clusters  
    - Demographic zones  
    - Crime density map.
- **Crime statistics charts (Chart.js)**  
  - Bar charts showing distribution of crime types near the queried location.  
  - Integrated into the chat page alongside the map.

---

## 3. Emergency & Safety Support Features

- **Emergency location sharing button**  
  - Fixed-position button on the map (bottom-right).
- **Automatic location retrieval**  
  - Uses the browser Geolocation API to obtain the users current coordinates.
- **Shareable emergency message**  
  - Generates a formatted emergency message including:  
    - Google Maps link  
    - Latitude/longitude values.  
  - Message is copied to clipboard for quick sharing via SMS, WhatsApp, etc.

---

## 4. Backend & API Features

- **Flask-based REST API** exposing:
  - `POST /api/chat` – text-based safety analysis for a location.  
  - `GET /api/heatmap` – crime heatmap data points.  
  - `GET /api/dbscan_clusters` – DBSCAN crime cluster data.  
  - `GET /api/demographic_zones` – demographic zone profiles.  
  - `GET /api/density_map` – crime density grid.
- **Central analysis pipeline**  
  - `analysis.py` computes safety scores, time-of-day effects, and contextual information for responses.
- **Geospatial utilities**  
  - `geo_utils.py` handles geocoding and distance calculations.
- **Data preprocessing and caching**  
  - `data_init.py` loads, preprocesses, and caches NYPD crime CSV data for efficient reuse.

---

## 5. Machine Learning & Analytics Features

- **Crime hotspot detection (DBSCAN)**  
  - Clusters spatial crime incidents to identify hotspots.  
  - Outputs cluster centers and dominant crime types.
- **Zone-based crime classification (KMeans)**  
  - Partitions the city into crime zones (e.g., 10 zones) based on location and crime features.
- **Demographic profiling**  
  - Uses KMeans on demographic + spatial features to create demographic zones (e.g., 25 zones).  
  - Analyzes victim characteristics (age, race, sex) by zone.
- **Crime density estimation (KDE)**  
  - Kernel Density Estimation on a grid (e.g., 50×50) over NYC.  
  - Classifies areas into low, medium, high crime density classes.
- **Safety score computation**  
  - Combines outputs from clustering, density, and demographic analysis to produce a single safety score.

---

## 6. Evaluation & Tooling Features

- **Model evaluation script (`model_evaluation.py`)**  
  - Computes clustering quality metrics:  
    - Silhouette score  
    - Davies–Bouldin index  
    - Calinski–Harabasz score.  
  - Analyzes zone classification performance and safety score consistency.  
  - Produces an overall system performance summary.

---

## 7. Frontend Application Features

- **ChatPage.jsx**  
  - Combines:  
    - Chat history  
    - Location query input  
    - Real-time safety response  
    - Integrated map and charts  
    - Emergency button.
- **SafetyMap.jsx**  
  - Renders the interactive Leaflet map with all overlays and markers.  
  - Ensures proper layout and responsiveness across devices.
- **GraphDisplay.jsx**  
  - Shows crime distribution charts (e.g., by crime type) for the selected area.

This list is intended as a high-level feature catalogue for documentation, pitching, and patent discussions.
