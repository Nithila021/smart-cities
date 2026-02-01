# 📊 Data Sources Documentation

This document provides links and instructions for downloading all datasets required to run the Smart Cities Crime Safety Analysis project.

---

## 🚨 Required Datasets

### 1. NYPD Crime Data (Primary Dataset)

**Source:** NYC Open Data  
**Update Frequency:** Daily

#### Option A: Current Year Data (Recommended for Development)
- **Dataset:** NYPD Complaint Data Current (Year To Date)
- **Link:** https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Current-Year-To-Date-/5uac-w243
- **File Name:** `NYPD_Complaint_Data_YTD.csv`
- **Size:** ~200MB (varies by year)
- **Records:** ~400,000+ complaints per year
- **Download:** Click "Export" → "CSV" button on the page

#### Option B: Historical Data (For Full Analysis)
- **Dataset:** NYPD Complaint Data Historic
- **Link:** https://data.cityofnewyork.us/Public-Safety/NYPD-Complaint-Data-Historic/qgea-i56i
- **File Name:** `NYPD_Complaint_Data_Historic.csv`
- **Size:** ~2GB+ (2006-2019 data)
- **Records:** 7+ million complaints
- **Download:** Click "Export" → "CSV" button on the page

**Installation:**
```bash
# Place the downloaded CSV file in the project root or backend/ directory
mv ~/Downloads/NYPD_Complaint_Data_YTD.csv /path/to/smart-cities-main/
# OR
mv ~/Downloads/NYPD_Complaint_Data_YTD.csv /path/to/smart-cities-main/backend/
```

**Key Columns Used:**
- `CMPLNT_FR_DT` - Complaint date
- `CMPLNT_FR_TM` - Complaint time
- `Latitude` / `Longitude` - Crime location
- `OFNS_DESC` - Offense description
- `LAW_CAT_CD` - Law category (Felony, Misdemeanor, Violation)
- `VIC_AGE_GROUP`, `VIC_SEX`, `VIC_RACE` - Victim demographics

---

## 🏙️ Optional Datasets (For Enhanced Analysis)

### 2. NYC 311 Service Requests

**Source:** NYC Open Data  
**Purpose:** Broken windows theory - quality of life indicators that correlate with crime

- **Dataset:** 311 Service Requests from 2010 to Present
- **Link:** https://data.cityofnewyork.us/Social-Services/311-Service-Requests-from-2010-to-Present/erm2-nwe9
- **File Name:** `311_Service_Requests.csv`
- **Size:** ~20GB+ (full dataset)
- **Records:** 30+ million requests
- **Recommended:** Filter by recent years or specific complaint types

**Useful Complaint Types:**
- Street Light Condition
- Noise - Street/Sidewalk
- Illegal Parking
- Graffiti
- Derelict Vehicles
- Homeless Encampment

**API Access:**
```bash
# Example: Get recent 311 complaints via Socrata API
curl "https://data.cityofnewyork.us/resource/erm2-nwe9.json?\$limit=1000&\$where=created_date>'2024-01-01'"
```

---

### 3. Points of Interest (POI) Data

**Source:** OpenStreetMap via Overpass API  
**Purpose:** Real amenity data (hospitals, schools, police stations, etc.)

**No download required** - Data is fetched dynamically via API

**Overpass API Endpoint:**
```
https://overpass-api.de/api/interpreter
```

**Example Query (Hospitals in NYC):**
```
[out:json];
(
  node["amenity"="hospital"](40.4774,-74.2591,40.9176,-73.7004);
  way["amenity"="hospital"](40.4774,-74.2591,40.9176,-73.7004);
);
out center;
```

**Supported Amenity Types:**
- `amenity=hospital` - Hospitals
- `amenity=school` - Schools
- `amenity=police` - Police stations
- `amenity=fire_station` - Fire stations
- `amenity=restaurant` - Restaurants
- `amenity=pharmacy` - Pharmacies
- `shop=supermarket` - Supermarkets

**Documentation:** https://wiki.openstreetmap.org/wiki/Overpass_API

---

### 4. NYC Street Lighting Data

**Source:** NYC Open Data  
**Purpose:** Poorly lit areas correlate with higher crime risk

- **Dataset:** NYC Street Centerline (CSCL)
- **Link:** https://data.cityofnewyork.us/City-Government/NYC-Street-Centerline-CSCL-/exjm-f27b
- **Alternative:** Street Light Locations
- **Link:** https://data.cityofnewyork.us/dataset/Street-Light-Locations/7ujv-9s5w

---

### 5. NYC Transit Data

**Source:** MTA GTFS  
**Purpose:** Subway/bus stops as congregation points

- **Dataset:** MTA GTFS Data
- **Link:** https://new.mta.info/developers
- **Format:** GTFS (General Transit Feed Specification)
- **Download:** http://web.mta.info/developers/data/nyct/subway/google_transit.zip

---

## 🗄️ Database Setup (PostgreSQL + PostGIS)

After downloading the NYPD data, you'll need to migrate it to PostgreSQL:

```bash
# 1. Install PostgreSQL 17 and PostGIS 3.6
brew install postgresql@17 postgis

# 2. Start PostgreSQL
brew services start postgresql@17

# 3. Create database
createdb -U postgres smart_cities

# 4. Enable PostGIS extension
psql -U postgres -d smart_cities -c "CREATE EXTENSION postgis;"

# 5. Run migration script (after schema is created)
cd backend
python migrate_data.py
```

---

## 📝 File Placement

Place downloaded files in the following locations:

```
smart-cities-main/
├── NYPD_Complaint_Data_YTD.csv          # Primary crime data
├── NYPD_Complaint_Data_Historic.csv     # Optional: Historical data
├── 311_Service_Requests.csv             # Optional: 311 data
└── backend/
    ├── schema.sql                        # Database schema (to be created)
    ├── migrate_data.py                   # Migration script (to be created)
    └── db_config.py                      # Database config (to be created)
```

---

## ⚠️ Important Notes

1. **File Size Warnings:**
   - NYPD Historic data is 2GB+ - ensure you have sufficient disk space
   - 311 data is 20GB+ - consider filtering before download

2. **API Rate Limits:**
   - Overpass API: 2 requests per second
   - NYC Open Data API: No strict limit, but be respectful

3. **Data Privacy:**
   - NYPD data has victim information redacted for privacy
   - Do not attempt to re-identify individuals

4. **Updates:**
   - NYPD data updates daily
   - 311 data updates daily
   - OSM data updates continuously

---

## 🔗 Additional Resources

- **NYC Open Data Portal:** https://opendata.cityofnewyork.us/
- **NYC Open Data API Docs:** https://dev.socrata.com/foundry/data.cityofnewyork.us/
- **OpenStreetMap Overpass API:** https://overpass-api.de/
- **PostGIS Documentation:** https://postgis.net/documentation/

---

**Last Updated:** 2026-02-01  
**Maintained By:** Smart Cities Team

