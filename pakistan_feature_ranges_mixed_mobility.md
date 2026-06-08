# Pakistan-Specific Feature Value Ranges (Mixed-Mobility Grid)
## For Synthetic EV Charging Dataset

> These ranges are based on real Pakistan EV market data (2025/2026), actual city coordinates, climate data, and the sociological shift towards 2-Wheeler EV Micro-Mobility.

---

## 🗺️ Reference: Major Pakistani Cities & Coordinates

| City | Latitude (°N) | Longitude (°E) | Climate Zone | Simulation Weight | Justification |
|------|--------------|----------------|--------------|-------------------|---------------|
| Karachi | 24.86 | 67.01 | Hot & Arid | **25%** (0.25) | Largest population, financial hub |
| Lahore | 31.55 | 74.35 | Semi-Arid, Monsoon | **20%** (0.20) | 2nd largest population |
| Islamabad | 33.72 | 73.04 | Humid Subtropical | **15%** (0.15) | High EV adoption, affluent demographic |
| Peshawar | 34.01 | 71.57 | Semi-Arid | **10%** (0.10) | Major northern hub, summer heat stress |
| Multan | 30.20 | 71.44 | Hot & Arid | **10%** (0.10) | Southern Punjab, extreme summer heat |
| Quetta | 30.18 | 67.00 | Highland/Cold Arid | **8%** (0.08) | Cold weather thermal derating test |
| Faisalabad | 31.42 | 73.09 | Hot Semi-Arid | **7%** (0.07) | High population but lower luxury EV adoption |
| Rawalpindi | 33.60 | 73.04 | Humid Subtropical | **5%** (0.05) | Supplements Islamabad (Twin Cities) |

---

## 📋 Feature-by-Feature Value Ranges

### Date_Time
- **Type:** Datetime
- **Values:** Random dates in 2026, aligned with Session_Start_Hour.
- **Justification:** Added to precisely match the real dataset's schema and preserve chronological alignment for time-series forecasting (TFT).

---

### Vehicle_ID
- **Type:** Integer
- **Values:** Sequential 1 to N
- **Latent Physics Note:** While the CSV only exports `Vehicle_ID` (to match the real dataset schema), the simulation internally generates a **Latent Vehicle Type** (80% 4-Wheelers, 20% 2-Wheelers) which governs physics calculations (Capacity, ECR, Load) without exposing a new column that would break the zero-shot pipeline.

---

### Battery_Capacity_kWh
- **Type:** Continuous (float)
- **Range:** `1.5 – 88.0 kWh`
- **Distribution:** Bimodal
- **Realistic Breakdown:**
  - 2-Wheeler Scooties (Evee C1, Metro T9, Jolta, Vlektra): 1.5 – 4.0 kWh → **40%** of fleet
  - Small/Budget EVs (Honri Ve, MG Binguo): 18.5 – 32 kWh → ~15% of fleet
  - Mid-range (ORA 03, Seres 3, MG ZS EV): 47 – 54 kWh → ~24% of fleet
  - High-end (Kia EV5, BYD Seal, Hyundai IONIQ): 64 – 88 kWh → ~21% of fleet
- **Justification:** Accurately reflects both the 2-wheeler micro-mobility boom and the current 4-wheeler market models in Pakistan 2026.

---

### State_of_Charge_%
- **Type:** Continuous (float)
- **Range:** `10 – 95 %`
- **Distribution:** Bimodal — peaks around 20–30% (initiating charge) and 80–90% (after charging)
- **Justification:** Drivers typically charge when SOC drops below 30% and stop at 80–90% to preserve battery health

---

### Energy_Consumption_Rate_kWh/km
- **Type:** Continuous (float)
- **Range:** `0.015 – 0.35 kWh/km`
- **Distribution:** Bimodal
- **Pakistan Context:**
  - 2-Wheelers: 0.015 – 0.05 kWh/km (Highly efficient, low impact from traffic)
  - 4-Wheelers (Highway/Good weather): 0.15 – 0.20 kWh/km
  - 4-Wheelers (Heavy traffic/AC load): 0.28 – 0.35 kWh/km
- **Justification:** Separates the massive efficiency of scooties from the high HVAC/traffic demands of 4-wheelers in Pakistan's hot climate.

---

### Current_Latitude
- **Type:** Continuous (float)
- **Range:** `24.5 – 34.5 °N`
- **Cities Used:** Karachi, Lahore, Islamabad, Peshawar, Quetta, Multan, Faisalabad, Rawalpindi
- **Distribution:** Uniform across city clusters with ±0.2° random jitter

---

### Current_Longitude
- **Type:** Continuous (float)
- **Range:** `67.0 – 74.5 °E`
- **Cities Used:** Same as above
- **Distribution:** Uniform across city clusters with ±0.2° random jitter

---

### Destination_Latitude
- **Type:** Continuous (float)
- **Range:** `24.5 – 34.5 °N`
- **Note:** Can be same city or different city (inter-city trips ~20% of records)
- **Distribution:** Same as Current Latitude

---

### Destination_Longitude
- **Type:** Continuous (float)
- **Range:** `67.0 – 74.5 °E`
- **Note:** Same as Destination Latitude logic

---

### Distance_to_Destination_km
- **Type:** Continuous (float)
- **Range:** `2 – 350 km`
- **Distribution:** Right-skewed; most trips are short urban trips
  - Urban (within city): 2–80 km → Covers 100% of 2-Wheelers and most 4-Wheelers
  - Inter-city: 100–350 km → Strictly 4-Wheelers
- **Justification:** Enforces physical limitations; 2-wheelers cannot realistically perform inter-city travel (e.g. Lahore to Islamabad).

---

### Traffic_Data
- **Type:** Integer
- **Range:** `50 – 3000 vehicles`
- **Distribution:** Right-skewed, correlated with Session Start Hour and Weekday
  - Peak hours (8–10 AM, 5–8 PM): 1500–3000
  - Off-peak (11 PM – 5 AM): 50–300
- **Pakistan Context:** Karachi and Lahore have among the worst traffic in South Asia

---

### Road_Conditions
- **Type:** Categorical
- **Values:** `Good`, `Average`, `Poor`
- **Distribution:**
  - Good: **20%** (major highways, new urban roads)
  - Average: **40%** (standard city roads)
  - Poor: **40%** (older city streets, rural routes)
- **Justification:** Pakistan's road infrastructure ranking is low; many roads are in poor/average condition, especially outside major urban centers

---

### Charging_Station_ID
- **Type:** Categorical (constant)
- **Value:** Same value for every row (e.g., `1` or `PKCS-001`)
- **Usage:** ❌ Not used as a model feature — kept only for structural consistency with the original dataset
- **Justification:** In your original dataset, Charging Station ID was uniform across all rows and excluded from model training. The synthetic dataset preserves this behavior.

---

### Charging_Rate_kW
- **Type:** Continuous (float)
- **Range:** `0.5 – 50 kW`
- **Distribution:**
  - Residential Wall Socket (0.5–1.5 kW): **40%** (2-Wheelers)
  - Slow AC (3.3–7.4 kW): **30%** (residential 4-wheelers)
  - Medium AC (11–22 kW): **21%** (commercial/mall charging)
  - Fast DC (22–50 kW): **9%** (highway/public fast chargers)
- **Justification:** Differentiates the low residential power draw of scooties from the massive commercial load of fast chargers.

---

### Queue_Time_mins
- **Type:** Continuous (float)
- **Range:** `0 – 90 mins`
- **Distribution:** Right-skewed
  - No queue (0 mins): ~40%
  - Short queue (1–20 mins): ~35%
  - Long queue (20–90 mins): ~25%
- **Pakistan Context:** Higher queue times due to very few charging stations; improves as infrastructure expands; correlated with Station Capacity

---

### Station_Capacity_EV
- **Type:** Integer
- **Range:** `1 – 8 EVs`
- **Distribution:**
  - 1–2 EVs: 35% (small/residential stations)
  - 3–4 EVs: 40% (commercial stations)
  - 5–8 EVs: 25% (large public/highway stations)
- **Justification:** Pakistan stations are typically small; dual-charger units (2×22 kW) are the norm for newer installations

---

### Time_Spent_Charging_mins
- **Type:** Continuous (float)
- **Range:** `20 – 300 mins`
- **Derived From:** `Energy Drawn (kWh) / Charging Rate (kW) × 60`
- **Distribution:** Right-skewed
  - Quick top-up (<60 mins): ~30%
  - Standard session (60–180 mins): ~50%
  - Full charge (180–300 mins): ~20%
- **Justification:** Slow AC chargers dominate, leading to longer session times

---

### Energy_Drawn_kWh
- **Type:** Continuous (float)
- **Range:** `0.2 – 70 kWh`
- **Derived From:** `Battery Capacity × (Target SOC – Current SOC) / 100`
- **Distribution:** Bimodal, heavily dependent on Vehicle Type
- **Justification:** 2-Wheelers will consistently draw less than 4kWh per session.

---

### Session_Start_Hour
- **Type:** Integer
- **Range:** `0 – 23`
- **Distribution:** Multi-modal
  - Morning peak (7–9 AM): ~20%
  - Afternoon (12–2 PM): ~15%
  - Evening peak (6–9 PM): ~30% ← **highest** (after work/commute)
  - Night (10 PM–12 AM): ~20%
  - Late night/early morning (1–6 AM): ~15%
- **Pakistan Context:** Evening charging is highest as workers return home; Friday afternoon dip due to Jumu'ah prayers

---

### Fleet_Size
- **Type:** Integer
- **Range:** `2 – 50 vehicles`
- **Distribution:** Right-skewed
  - Small fleet (2–10): 55%
  - Medium fleet (11–30): 35%
  - Large fleet (31–50): 10%
- **Justification:** Pakistan's EV fleet market is nascent; mostly small commercial/rideshare fleets (Kareem EV pilots, corporate fleets)

---

### Fleet_Schedule
- **Type:** Binary (Integer)
- **Values:** `0 = On Time`, `1 = Delayed`
- **Distribution:** 0: **55%**, 1: **45%**
- **Pakistan Context:** Higher delay rate (~45%) due to traffic congestion, poor road conditions, and infrastructure challenges

---

### Temperature_C
- **Type:** Continuous (float)
- **Range:** `-2 – 50 °C`
- **Distribution by Region/Season:**
  - Karachi: 20–45°C (hot year-round)
  - Lahore: 5–48°C (extreme summers, mild winters)
  - Islamabad/Rawalpindi: 2–42°C
  - Peshawar: 3–45°C
  - Quetta: -2–38°C (coldest major city)
- **Justification:** Pakistan has extreme temperature ranges; heat significantly increases AC energy load, cold reduces battery efficiency

---

### Wind_Speed_m/s
- **Type:** Continuous (float)
- **Range:** `0.5 – 18 m/s`
- **Distribution:** Right-skewed, mean ≈ 4 m/s
  - Calm (0.5–3 m/s): 50%
  - Moderate (3–8 m/s): 35%
  - Strong (8–18 m/s): 15% (coastal Karachi, Balochistan winds)
- **Justification:** Karachi is coastal and windy; Balochistan has strong seasonal winds

---

### Precipitation_mm
- **Type:** Continuous (float)
- **Range:** `0 – 80 mm`
- **Distribution:** Heavily right-skewed
  - No rain (0 mm): **65%** (Pakistan is predominantly arid)
  - Light rain (0.1–10 mm): **20%**
  - Moderate rain (10–40 mm): **10%** (Monsoon Jul–Sep)
  - Heavy rain (40–80 mm): **5%** (Karachi/Lahore flash floods)
- **Justification:** Pakistan receives most rainfall during monsoon season (July–September); vast areas (Balochistan, Sindh interior) are very dry year-round

---

### Weekday
- **Type:** Integer
- **Range:** `0 – 6`  (0 = Monday, 6 = Sunday)
- **Distribution:** Roughly uniform with slight variation
  - Weekdays (Mon–Thu, 0–3): higher charging activity
  - Friday (4): slight dip mid-day (Jumu'ah prayers), active morning & evening
  - Weekend (Sat–Sun, 5–6): lower overall, but evening charging peak remains
- **Pakistan Context:** Pakistan's weekend is Saturday-Sunday; Friday has cultural significance affecting midday behavior

---

### Charging_Preferences
- **Type:** Binary (Integer)
- **Values:** `0 = No Preference`, `1 = Has Preference`
- **Distribution:** 0: **65%**, 1: **35%**
- **Pakistan Context:** Many EV users are new adopters; fewer users have established charging preferences compared to mature markets; those with preference usually select nearby/cheaper stations

---

### Weather_Conditions
- **Type:** Categorical
- **Values:** `Clear`, `Cloudy`, `Rain`, `Storm`
- **Distribution:**
  - Clear: **55%** (dominant; Pakistan is mostly sunny)
  - Cloudy: **25%**
  - Rain: **17%** (mainly monsoon season)
  - Storm: **3%** (rare; Karachi urban flooding events)
- **Justification:** Pakistan has ~300 sunny days/year in most regions; storms are rare

---

### Charging_Load_kW ← Target Variable
- **Type:** Continuous (float)
- **Range:** `0.4 – 50 kW`
- **Derived From:** Combination of Vehicle Type, Charging Rate, SOC, Temperature, Traffic conditions
- **Distribution:** Bimodal — residential peaks around 0.5–1.5 kW (Scooties) and commercial peaks around 7–50 kW (Cars)
- **Key Influencing Factors:**
  - **2-Wheelers have NO HVAC overhead power draw** (locked to 0.0 even in extreme heat).
  - High temperature → higher load for 4-wheelers (AC cooling during charging).
  - Fast chargers → high spike loads.
- **Justification:** This is the target label. It is physically rigorous, separating residential micro-mobility from commercial EV charging loads.
