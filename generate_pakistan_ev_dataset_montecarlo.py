"""
Monte Carlo Synthetic EV Charging Dataset Generator — Pakistan
==============================================================
Approach: Monte Carlo Simulation

What makes this Monte Carlo:
  1. Every uncertain input is modelled with a proper parametric distribution
     (Beta, Truncated-Normal, Weibull, Gamma, Log-Normal, etc.) rather than
     simple uniform/normal draws.
  2. For each simulation trial (row) all inputs are sampled simultaneously
     from their distributions.
  3. Dependent / secondary variables are DERIVED by propagating the sampled
     inputs through physics-based engineering models.
  4. The target variable (Charging Load) emerges as the OUTPUT of the
     stochastic physical model — not sampled directly.

Distributions Used:
  - Beta          : SOC %, charger efficiency, road-condition coefficient
  - Truncated-Normal : Temperature, Energy Consumption Rate
  - Weibull       : Wind Speed  (standard in wind modelling)
  - Gamma         : Queue Time  (standard for waiting-time modelling)
  - Log-Normal    : Traffic Data, Fleet Size (right-skewed counts)
  - Multinomial   : City choice, Session Start Hour, Weekday
  - Bernoulli     : Fleet Schedule, Charging Preferences
  - Mixture       : Battery Capacity (3-segment), Charging Rate (3-segment)

Physical Model for Charging Load:
  Charging Load (kW) = Charging_Rate
                       × η_thermal(T)         # thermal derating
                       × η_charger            # charger efficiency (Beta-sampled)
                       + P_hvac               # HVAC overhead at extreme temps

Author : Generated for Pakistan Synthetic EV Dataset — Thesis
Seed   : 42  (reproducible)
"""

import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# CONFIGURATION
# ============================================================
N_SAMPLES   = 50000
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

OUTPUT_PATH = r'd:\NED\Thesis\Synthesized EV Data\Pakistan_EV_Charging_Dataset_MonteCarlo.csv'

# ============================================================
# Pakistani Cities — coordinates + truncated-normal temp params
# ============================================================
CITIES = {
    'Karachi':    {'lat': 24.86, 'lon': 67.01, 'temp_mean': 32, 'temp_std':  7, 'temp_min': 20, 'temp_max': 45, 'weight': 0.25},
    'Lahore':     {'lat': 31.55, 'lon': 74.35, 'temp_mean': 25, 'temp_std': 12, 'temp_min':  5, 'temp_max': 48, 'weight': 0.20},
    'Islamabad':  {'lat': 33.72, 'lon': 73.04, 'temp_mean': 22, 'temp_std': 11, 'temp_min':  2, 'temp_max': 42, 'weight': 0.15},
    'Peshawar':   {'lat': 34.01, 'lon': 71.57, 'temp_mean': 24, 'temp_std': 12, 'temp_min':  3, 'temp_max': 45, 'weight': 0.10},
    'Quetta':     {'lat': 30.18, 'lon': 67.00, 'temp_mean': 18, 'temp_std': 13, 'temp_min': -2, 'temp_max': 38, 'weight': 0.08},
    'Multan':     {'lat': 30.20, 'lon': 71.44, 'temp_mean': 28, 'temp_std': 12, 'temp_min':  7, 'temp_max': 48, 'weight': 0.10},
    'Faisalabad': {'lat': 31.42, 'lon': 73.09, 'temp_mean': 26, 'temp_std': 12, 'temp_min':  5, 'temp_max': 47, 'weight': 0.07},
    'Rawalpindi': {'lat': 33.60, 'lon': 73.04, 'temp_mean': 22, 'temp_std': 11, 'temp_min':  2, 'temp_max': 43, 'weight': 0.05},
}
city_names   = list(CITIES.keys())
city_weights = np.array([CITIES[c]['weight'] for c in city_names])
city_weights /= city_weights.sum()

# ============================================================
# PHYSICS-BASED CHARGING LOAD MODEL
# ============================================================
def thermal_efficiency(temperature):
    """
    Thermal derating model for EV battery charging.
    Based on standard lithium-ion thermal behaviour:
      - Optimal range  10–30 deg C  → efficiency = 1.0
      - Above  30 deg C             → linear derating (heat stress)
      - Below  10 deg C             → linear derating (cold resistance)
    Returns a per-sample efficiency vector in [0.80, 1.00].
    """
    eta = np.ones_like(temperature, dtype=float)
    hot_mask  = temperature > 30
    cold_mask = temperature < 10
    eta[hot_mask]  = 1.0 - 0.006 * (temperature[hot_mask]  - 30)   # -0.6% per degC above 30
    eta[cold_mask] = 1.0 - 0.005 * (10 - temperature[cold_mask])   # -0.5% per degC below 10
    return np.clip(eta, 0.80, 1.00)

def hvac_overhead(temperature):
    """
    Additional power drawn by climate control system (kW).
    High heat or cold both activate HVAC.
    """
    overhead = np.zeros_like(temperature, dtype=float)
    overhead[temperature > 35] = np.random.uniform(0.8, 1.5, np.sum(temperature > 35))
    overhead[temperature < 5]  = np.random.uniform(0.5, 1.0, np.sum(temperature < 5))
    return overhead

# ============================================================
# MONTE CARLO SIMULATION — STEP BY STEP
# ============================================================

print("Running Monte Carlo simulation...")
print(f"  Trials (rows) : {N_SAMPLES:,}")
print(f"  Features       : 27  (26 input + 1 target)")
print()

# ------------------------------------------------------------------
# [MC-1] Assign city per trial  — Multinomial draw
# ------------------------------------------------------------------
city_idx    = np.random.choice(len(city_names), size=N_SAMPLES, p=city_weights)
current_city = [city_names[i] for i in city_idx]
print("  [1/26] City assignment done")

# ------------------------------------------------------------------
# [MC-2] Vehicle ID & Vehicle Type
# ------------------------------------------------------------------
vehicle_id = np.arange(1, N_SAMPLES + 1)
vehicle_type = np.random.choice(['4-Wheeler', '2-Wheeler'], size=N_SAMPLES, p=[0.80, 0.20])
print("  [1b/26] Vehicle Type sampled     (4-Wheeler 80%, 2-Wheeler 20%)")

# ------------------------------------------------------------------
# [MC-3] Battery Capacity (kWh)
# Distribution: 3-component mixture of Truncated-Normals
#   Segment 0 Budget   [18.5, 32.0]  weight=0.25
#   Segment 1 Mid      [47.0, 54.0]  weight=0.40
#   Segment 2 High-end [64.0, 88.0]  weight=0.35
# ------------------------------------------------------------------
def sample_truncnorm(mean, std, lo, hi, n):
    a, b = (lo - mean) / std, (hi - mean) / std
    return stats.truncnorm.rvs(a, b, loc=mean, scale=std, size=n)

seg = np.random.choice([0, 1, 2], size=N_SAMPLES, p=[0.25, 0.40, 0.35])
cap_4w = np.where(
    seg == 0, sample_truncnorm(25.0,  4.5, 18.5, 32.0, N_SAMPLES),
    np.where(
        seg == 1, sample_truncnorm(51.0,  3.0, 47.0, 54.0, N_SAMPLES),
                  sample_truncnorm(75.0, 10.0, 64.0, 88.0, N_SAMPLES)
    )
)
cap_2w = sample_truncnorm(2.5, 0.8, 1.5, 4.0, N_SAMPLES)
battery_capacity = np.where(vehicle_type == '4-Wheeler', cap_4w, cap_2w)
battery_capacity = np.round(battery_capacity, 1)
print("  [2/26] Battery Capacity sampled  (Truncated-Normal mixture)")

# ------------------------------------------------------------------
# [MC-4] State of Charge (SOC %)
# Distribution: Right-skewed Beta
# Justification: Vehicles arriving at a charging station mostly have low SOC.
#   Uses Beta(2, 5) scaled to [5, 60] -> peak is around 15-25%
# ------------------------------------------------------------------
soc = stats.beta.rvs(2, 5, size=N_SAMPLES) * 55 + 5   # [5, 60]
# Add a very small chance (5%) of "top-up" charging at higher SOC [60, 80]
top_up_mask = np.random.rand(N_SAMPLES) < 0.05
soc[top_up_mask] = np.random.uniform(60, 80, np.sum(top_up_mask))
soc = np.round(np.clip(soc, 5, 85), 1)
print("  [3/26] SOC sampled               (Beta mixture)")

# ------------------------------------------------------------------
# [MC-5] Session Start Hour — Multinomial over 24 hours
# ------------------------------------------------------------------
hour_probs = np.array([
    0.015, 0.010, 0.008, 0.008, 0.010, 0.015,   # 00-05
    0.025, 0.060, 0.080, 0.060, 0.040, 0.040,   # 06-11
    0.050, 0.055, 0.040, 0.040, 0.045, 0.060,   # 12-17
    0.090, 0.100, 0.095, 0.070, 0.050, 0.030,   # 18-23
])
hour_probs /= hour_probs.sum()
session_start_hour = np.random.choice(np.arange(24), size=N_SAMPLES, p=hour_probs)
print("  [4/26] Session Start Hour sampled (Multinomial)")

# ------------------------------------------------------------------
# [MC-6] Weekday — Multinomial (Fri lighter midday; Sat-Sun lighter)
# ------------------------------------------------------------------
weekday = np.random.choice(7, size=N_SAMPLES, p=[0.16, 0.16, 0.16, 0.16, 0.14, 0.11, 0.11])
print("  [5/26] Weekday sampled           (Multinomial)")

# ------------------------------------------------------------------
# [MC-7] Temperature (°C)
# Distribution: Truncated-Normal — city-specific mean & std
# ------------------------------------------------------------------
temp_mean = np.array([CITIES[c]['temp_mean'] for c in current_city], dtype=float)
temp_std  = np.array([CITIES[c]['temp_std']  for c in current_city], dtype=float)
temp_lo   = np.array([CITIES[c]['temp_min']  for c in current_city], dtype=float)
temp_hi   = np.array([CITIES[c]['temp_max']  for c in current_city], dtype=float)
a_t = (temp_lo - temp_mean) / temp_std
b_t = (temp_hi - temp_mean) / temp_std
temperature = stats.truncnorm.rvs(a_t, b_t, loc=temp_mean, scale=temp_std)
temperature = np.round(temperature, 1)
print("  [6/26] Temperature sampled       (Truncated-Normal, city-specific)")

# ------------------------------------------------------------------
# [MC-8] Road Conditions — categorical with Beta-sampled threshold
# Uses a Beta(2,2) latent variable → thresholds produce 20/40/40 split
# ------------------------------------------------------------------
road_latent = stats.beta.rvs(2, 2, size=N_SAMPLES)
road_conditions = np.where(road_latent > 0.80, 'Good',
                  np.where(road_latent > 0.40, 'Average', 'Poor'))
print("  [7/26] Road Conditions sampled   (Beta latent variable)")

# ------------------------------------------------------------------
# [MC-9] Traffic Data
# Distribution: Log-Normal — peak/off-peak mu & sigma differ
# ------------------------------------------------------------------
peak_mask    = ((session_start_hour >= 7)  & (session_start_hour <= 9))  | \
               ((session_start_hour >= 17) & (session_start_hour <= 20))
offpeak_mask = (session_start_hour >= 23) | (session_start_hour <= 5)
weekend_mask = (weekday == 5) | (weekday == 6)

# Log-normal parameters (mu, sigma) for log-space
lnorm_mu    = np.where(peak_mask, 7.4,   np.where(offpeak_mask, 5.0, 6.8))
lnorm_sigma = np.full(N_SAMPLES, 0.5)
lnorm_mu    = np.where(weekend_mask, lnorm_mu - 0.4, lnorm_mu)   # reduce for weekend

traffic_data = stats.lognorm.rvs(s=lnorm_sigma, scale=np.exp(lnorm_mu))
traffic_data = np.clip(traffic_data, 50, 3000).astype(int)
print("  [8/26] Traffic Data sampled      (Log-Normal, hour/weekday conditioned)")

# ------------------------------------------------------------------
# [MC-10 & 11] Current Latitude / Longitude  (city centred + jitter)
# ------------------------------------------------------------------
current_lat = np.array([CITIES[c]['lat'] for c in current_city]) + \
              stats.truncnorm.rvs(-2, 2, scale=0.10, size=N_SAMPLES)
current_lon = np.array([CITIES[c]['lon'] for c in current_city]) + \
              stats.truncnorm.rvs(-2, 2, scale=0.10, size=N_SAMPLES)
current_lat = np.round(current_lat, 4)
current_lon = np.round(current_lon, 4)
print("  [9/26] Current Lat/Lon sampled   (Truncated-Normal jitter around city)")

# ------------------------------------------------------------------
# [MC-12 & 13] Destination Latitude / Longitude
# 70% within same city, 30% inter-city
# ------------------------------------------------------------------
inter_city   = np.random.rand(N_SAMPLES) < 0.30
dest_idx     = np.where(inter_city,
                   np.random.choice(len(city_names), size=N_SAMPLES, p=city_weights),
                   city_idx)
dest_city    = [city_names[i] for i in dest_idx]
dest_lat     = np.array([CITIES[c]['lat'] for c in dest_city]) + \
               stats.truncnorm.rvs(-2, 2, scale=0.10, size=N_SAMPLES)
dest_lon     = np.array([CITIES[c]['lon'] for c in dest_city]) + \
               stats.truncnorm.rvs(-2, 2, scale=0.10, size=N_SAMPLES)
dest_lat     = np.round(dest_lat, 4)
dest_lon     = np.round(dest_lon, 4)
print("  [10/26] Destination Lat/Lon sampled")

# ------------------------------------------------------------------
# [MC-14] Distance to Destination (km)
# Within-city : Gamma(shape=2, scale=8)  → right-skewed short distances
# Inter-city  : Truncated-Normal [100, 350]
# ------------------------------------------------------------------
same_city       = np.array(dest_city) == np.array(current_city)
dist_urban      = stats.gamma.rvs(a=2, scale=8, size=N_SAMPLES) + 5     # [5, ~80]
dist_intercity  = sample_truncnorm(200, 65, 100, 350, N_SAMPLES)
# Physics rule: 2-Wheelers realistically do not do long inter-city travel
distance        = np.where((same_city) | (vehicle_type == '2-Wheeler'), dist_urban, dist_intercity)
distance        = np.round(np.clip(distance, 2, 350), 1)
print("  [11/26] Distance to Destination sampled (Gamma / Truncated-Normal)")

# ------------------------------------------------------------------
# [MC-15] Energy Consumption Rate (kWh/km)
# Distribution: Truncated-Normal, mean shifted by road/temp/traffic
# ------------------------------------------------------------------
base_mean   = 0.22
road_pen    = np.where(road_conditions == 'Poor', 0.04,
              np.where(road_conditions == 'Average', 0.02, 0.0))
temp_pen    = np.where(temperature > 35, 0.03, np.where(temperature < 5, 0.02, 0.0))
traffic_pen = np.where(traffic_data > 1500, 0.03, np.where(traffic_data > 800, 0.01, 0.0))
ecr_mean    = base_mean + road_pen + temp_pen + traffic_pen
ecr_4w = stats.truncnorm.rvs(
    (0.15 - ecr_mean) / 0.025,
    (0.35 - ecr_mean) / 0.025,
    loc=ecr_mean, scale=0.025
)
# Physics rule: 2-Wheelers consume massively less energy and are less hindered by traffic
ecr_2w_mean = 0.03 + (road_pen * 0.1) + (traffic_pen * 0.1)
ecr_2w = stats.truncnorm.rvs((0.015 - ecr_2w_mean) / 0.005, (0.05 - ecr_2w_mean) / 0.005, loc=ecr_2w_mean, scale=0.005)

ecr = np.where(vehicle_type == '4-Wheeler', ecr_4w, ecr_2w)
ecr = np.round(np.clip(ecr, 0.015, 0.35), 3)
print("  [12/26] Energy Consumption Rate sampled (Truncated-Normal, conditioned)")

# ------------------------------------------------------------------
# [MC-16] Charging Rate (kW)
# Distribution: 3-component mixture of Truncated-Normals
# ------------------------------------------------------------------
rate_seg = np.random.choice([0, 1, 2], size=N_SAMPLES, p=[0.50, 0.35, 0.15])
rate_4w = np.where(
    rate_seg == 0, sample_truncnorm( 5.5,  1.2,   3.3,  7.4, N_SAMPLES),
    np.where(
        rate_seg == 1, sample_truncnorm(16.0,  3.5,  11.0, 22.0, N_SAMPLES),
                       sample_truncnorm(35.0,  9.0,  22.0, 50.0, N_SAMPLES)
    )
)
# Physics rule: 2-Wheelers charge via wall socket (0.5 to 1.5 kW)
rate_2w = sample_truncnorm(1.0, 0.2, 0.5, 1.5, N_SAMPLES)
charging_rate = np.where(vehicle_type == '4-Wheeler', rate_4w, rate_2w)
charging_rate = np.round(np.clip(charging_rate, 0.5, 50.0), 1)
print("  [13/26] Charging Rate sampled    (Truncated-Normal mixture)")

# ------------------------------------------------------------------
# [MC-17] Station Capacity (EVs) — Discrete Multinomial
# ------------------------------------------------------------------
cap_seg = np.random.choice([0, 1, 2], size=N_SAMPLES, p=[0.35, 0.40, 0.25])
station_capacity = np.where(
    cap_seg == 0, np.random.randint(1, 3, N_SAMPLES),
    np.where(cap_seg == 1, np.random.randint(3, 5, N_SAMPLES),
                           np.random.randint(5, 9, N_SAMPLES))
)
print("  [14/26] Station Capacity sampled (Discrete Multinomial)")

# ------------------------------------------------------------------
# [MC-18] Fleet Size
# Distribution: Log-Normal (right-skewed counts)
# ------------------------------------------------------------------
fleet_lnorm = stats.lognorm.rvs(s=0.9, scale=np.exp(2.3), size=N_SAMPLES)
fleet_size  = np.clip(fleet_lnorm.astype(int), 2, 50)
print("  [15/26] Fleet Size sampled       (Log-Normal)")

# ------------------------------------------------------------------
# [MC-19] Queue Time (mins)
# Distribution: Gamma — shape & scale driven by congestion ratio
# ------------------------------------------------------------------
congestion  = fleet_size / np.maximum(station_capacity, 1)
gamma_shape = np.where(congestion <= 3, 1.0, np.where(congestion <= 8, 2.0, 3.5))
gamma_scale = np.where(congestion <= 3, 3.0, np.where(congestion <= 8, 8.0, 12.0))
queue_time  = stats.gamma.rvs(a=gamma_shape, scale=gamma_scale)
queue_time  = np.round(np.clip(queue_time, 0, 90), 1)
print("  [16/26] Queue Time sampled       (Gamma, congestion-conditioned)")

# ------------------------------------------------------------------
# [MC-20] Energy Drawn (kWh)  — Derived via physical formula
# target_soc sampled from Beta(3,2) to model partial charge preference
# ------------------------------------------------------------------
charge_amount = stats.beta.rvs(3, 2, size=N_SAMPLES) * 60 + 15   # 15–75% top-up
target_soc    = np.clip(soc + charge_amount, 50, 95)
energy_drawn  = battery_capacity * (target_soc - soc) / 100
energy_drawn  = np.where(vehicle_type == '4-Wheeler', np.clip(energy_drawn, 3, 70), np.clip(energy_drawn, 0.2, 4.0))
energy_drawn  = np.round(energy_drawn, 2)
print("  [17/26] Energy Drawn derived     (Physical: dE = Cap x dSOC/100)")

# ------------------------------------------------------------------
# [MC-21] Time Spent Charging (mins)  — Derived
# ------------------------------------------------------------------
time_spent_charging = np.clip((energy_drawn / charging_rate) * 60, 20, 300)
time_spent_charging = np.round(time_spent_charging, 1)
print("  [18/26] Time Spent Charging derived  (Physical: t = E/P x 60)")

# ------------------------------------------------------------------
# [MC-22] Fleet Schedule — Bernoulli
# ------------------------------------------------------------------
fleet_schedule = stats.bernoulli.rvs(p=0.45, size=N_SAMPLES)   # 45% delayed
print("  [19/26] Fleet Schedule sampled   (Bernoulli)")

# ------------------------------------------------------------------
# [MC-23] Wind Speed (m/s)
# Distribution: Weibull  (industry standard for wind)
# shape=1.8 (typical for South Asian cities)
# ------------------------------------------------------------------
wind_speed = stats.weibull_min.rvs(c=1.8, scale=4.5, size=N_SAMPLES)
wind_speed = np.round(np.clip(wind_speed, 0.5, 18.0), 1)
print("  [20/26] Wind Speed sampled       (Weibull, k=1.8)")

# ------------------------------------------------------------------
# [MC-24] Weather Conditions — Multinomial
# ------------------------------------------------------------------
weather_conditions = np.random.choice(
    ['Clear', 'Cloudy', 'Rain', 'Storm'],
    size=N_SAMPLES, p=[0.55, 0.25, 0.17, 0.03]
)
print("  [21/26] Weather Conditions sampled (Multinomial)")

# ------------------------------------------------------------------
# [MC-25] Precipitation (mm)
# Distribution: Gamma conditioned on weather category
#   Clear  → 0
#   Cloudy → Gamma(1.0, 1.5)  [0–~10 mm]
#   Rain   → Gamma(2.5, 6.0)  [5–~40 mm]
#   Storm  → Gamma(4.0, 12.0) [20–~80 mm]
# ------------------------------------------------------------------
precip_clear   = np.zeros(N_SAMPLES)
precip_cloudy  = stats.gamma.rvs(a=1.0, scale=1.5, size=N_SAMPLES)
precip_rain    = stats.gamma.rvs(a=2.5, scale=6.0, size=N_SAMPLES)
precip_storm   = stats.gamma.rvs(a=4.0, scale=12.0, size=N_SAMPLES)

precipitation = np.where(weather_conditions == 'Storm',  precip_storm,
               np.where(weather_conditions == 'Rain',    precip_rain,
               np.where(weather_conditions == 'Cloudy',  precip_cloudy,
                        precip_clear)))
precipitation = np.round(np.clip(precipitation, 0, 80), 1)
print("  [22/26] Precipitation sampled    (Gamma, weather-conditioned)")

# ------------------------------------------------------------------
# [MC-26] Charging Preferences — Bernoulli
# ------------------------------------------------------------------
charging_preferences = stats.bernoulli.rvs(p=0.35, size=N_SAMPLES)
print("  [23/26] Charging Preferences sampled (Bernoulli)")

# ------------------------------------------------------------------
# Charging Station ID — constant (not a model feature)
# ------------------------------------------------------------------
charging_station_id = np.ones(N_SAMPLES, dtype=int)

# ------------------------------------------------------------------
# [MC-27] CHARGING LOAD (kW)  ← TARGET VARIABLE
# -------------------------------------------------------
# Derived via physics-based stochastic model:
#
#   Charging Load = Charging_Rate
#                   × η_thermal(T)         [thermal derating]
#                   × η_charger            [charger efficiency — Beta sampled]
#                   + P_hvac(T)            [HVAC overhead — stochastic]
#
# This is the core of the Monte Carlo approach:
# uncertainty in η_charger and P_hvac propagates into the output.
# ------------------------------------------------------------------
eta_thermal  = thermal_efficiency(temperature)

# Charger efficiency: Beta(8,2) → mean ~0.80, range [0.70, 0.98]
eta_charger  = stats.beta.rvs(8, 2, size=N_SAMPLES)
eta_charger  = np.clip(eta_charger, 0.70, 0.98)

p_hvac       = hvac_overhead(temperature)
# Physics rule: 2-Wheelers do not have HVAC systems!
p_hvac       = np.where(vehicle_type == '2-Wheeler', 0.0, p_hvac)

charging_load = charging_rate * eta_thermal * eta_charger + p_hvac
charging_load = np.where(vehicle_type == '4-Wheeler', 
                         np.clip(charging_load, 3.3, 50.0),
                         np.clip(charging_load, 0.4, 2.0))
charging_load = np.round(charging_load, 2)
print("  [24/26] Charging Load computed   (Physics model: P = Rate x eta_T x eta_C + P_HVAC)")

# ============================================================
# ASSEMBLE DATAFRAME  (EXACT column names and order of original dataset)
# ============================================================

# Create a Date_Time column that physically aligns with the session_start_hour
# We randomly assign days in 2026, but the HOUR perfectly matches the Monte Carlo distribution!
random_days = np.random.choice(pd.date_range('2026-01-01', '2026-12-31').strftime('%Y-%m-%d'), size=N_SAMPLES)
date_time_str = [f"{day} {hour:02d}:00:00" for day, hour in zip(random_days, session_start_hour)]
date_time = pd.to_datetime(date_time_str)

df = pd.DataFrame({
    'Date_Time':                        date_time,
    'Vehicle_ID':                       vehicle_id,
    'Battery_Capacity_kWh':             battery_capacity,
    'State_of_Charge_%':                soc,
    'Energy_Consumption_Rate_kWh/km':   ecr,
    'Current_Latitude':                 current_lat,
    'Current_Longitude':                current_lon,
    'Destination_Latitude':             dest_lat,
    'Destination_Longitude':            dest_lon,
    'Distance_to_Destination_km':       distance,
    'Traffic_Data':                     traffic_data,
    'Road_Conditions':                  road_conditions,
    'Charging_Station_ID':              charging_station_id,
    'Charging_Rate_kW':                 charging_rate,
    'Queue_Time_mins':                  queue_time,
    'Station_Capacity_EV':              station_capacity,
    'Time_Spent_Charging_mins':         time_spent_charging,
    'Energy_Drawn_kWh':                 energy_drawn,
    'Session_Start_Hour':               session_start_hour,
    'Fleet_Size':                       fleet_size,
    'Fleet_Schedule':                   fleet_schedule,
    'Temperature_C':                    temperature,
    'Wind_Speed_m/s':                   wind_speed,
    'Precipitation_mm':                 precipitation,
    'Weekday':                          weekday,
    'Charging_Preferences':             charging_preferences,
    'Weather_Conditions':               weather_conditions,
    'Charging_Load_kW':                 charging_load,
})

# ============================================================
# SAVE
# ============================================================
# Sort by Date_Time to ensure chronological order for time-series forecasting (TFT)
df.sort_values(by='Date_Time', inplace=True)
df.reset_index(drop=True, inplace=True)

df.to_csv(OUTPUT_PATH, index=False)

print()
print("=" * 65)
print("Pakistan EV Charging Dataset (Monte Carlo) -- Generated!")
print("=" * 65)
print(f"   Saved to : {OUTPUT_PATH}")
print(f"   Shape    : {df.shape[0]:,} rows x {df.shape[1]} columns")
print()
print("-" * 65)
print("Numerical Feature Summary:")
print("-" * 65)
print(df.describe().round(2).to_string())
print()
print("-" * 65)
print("Categorical Value Counts:")
print("-" * 65)
print("\n  Road Conditions:")
print(df['Road_Conditions'].value_counts().to_string(header=False))
print("\n  Weather Conditions:")
print(df['Weather_Conditions'].value_counts().to_string(header=False))
print("\n  Fleet Schedule (0=On Time, 1=Delayed):")
print(df['Fleet_Schedule'].value_counts().to_string(header=False))
print("\n  Charging Preferences (0=No, 1=Yes):")
print(df['Charging_Preferences'].value_counts().to_string(header=False))
print()
print("-" * 65)
print("Charging Load (kW) -- Target Variable Stats:")
print("-" * 65)
print(f"   Min    : {df['Charging_Load_kW'].min()}")
print(f"   Max    : {df['Charging_Load_kW'].max()}")
print(f"   Mean   : {df['Charging_Load_kW'].mean():.3f}")
print(f"   Median : {df['Charging_Load_kW'].median():.3f}")
print(f"   Std    : {df['Charging_Load_kW'].std():.3f}")
print("=" * 65)
