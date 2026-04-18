import streamlit as st
import pandas as pd
import requests
import json
import re
import base64
import folium
import plotly.express as px
from datetime import datetime
from streamlit_folium import st_folium
from databricks.sdk import WorkspaceClient
from openai import OpenAI
from geopy.distance import geodesic

# --- 1. INITIALIZATION & CONFIG ---
w = WorkspaceClient()
WAREHOUSE_ID = "2c3b173ddf7eb7ca"
ANALYTICS_TABLE = "workspace.default.route_spoilage_analytics"
CROP_REF_TABLE = "workspace.default.crop_reference_data"

# ── SARVAM CONFIG ──────────────────────────────────────────────────
SARVAM_API_KEY = "sk_4vap260f_hIWj3AZ6cyVmdhSekuYxth1k"
SARVAM_HEADERS = {
    "api-subscription-key": SARVAM_API_KEY,
    "Content-Type": "application/json"
}
llm_client = OpenAI(
    base_url="https://api.sarvam.ai/v1",
    api_key=SARVAM_API_KEY
).with_options(max_retries=1)

st.set_page_config(page_title="KrishiMarg : Route Risk Analyzer", layout="wide")

if 'current_sim' not in st.session_state:
    st.session_state['current_sim'] = None
if 'ai_recommendation' not in st.session_state:
    st.session_state['ai_recommendation'] = None

# --- 2. DATA LOADERS ---
@st.cache_data(ttl=30)
def fetch_history():
    try:
        query = f"SELECT * FROM {ANALYTICS_TABLE} ORDER BY runtime DESC"
        res = w.statement_execution.execute_statement(
            warehouse_id=WAREHOUSE_ID, statement=query, wait_timeout="30s"
        )
        if res.result is None or res.result.data_array is None:
            return pd.DataFrame()
        cols = [c.name.lower() for c in res.manifest.schema.columns]
        df = pd.DataFrame(res.result.data_array, columns=cols)
        num_cols = ['risk_score', 'time_hr', 'dist_km', 'avg_temp', 'max_temp', 'min_temp', 'avg_humidity']
        existing = [c for c in num_cols if c in df.columns]
        df[existing] = df[existing].apply(pd.to_numeric, errors='coerce')
        return df
    except Exception as e:
        st.warning(f"Could not load history: {e}")
        return pd.DataFrame()

@st.cache_data
def load_crops():
    try:
        res = w.statement_execution.execute_statement(
            warehouse_id=WAREHOUSE_ID,
            statement=f"SELECT * FROM {CROP_REF_TABLE}",
            wait_timeout="30s"
        )
        if res.result and res.result.data_array:
            cols = [c.name.lower() for c in res.manifest.schema.columns]
            return pd.DataFrame(res.result.data_array, columns=cols).set_index('crop').to_dict('index')
    except:
        pass
    try:
        res = w.statement_execution.execute_statement(
            warehouse_id=WAREHOUSE_ID,
            statement=f"SELECT DISTINCT crop FROM {ANALYTICS_TABLE} WHERE crop IS NOT NULL",
            wait_timeout="30s"
        )
        if res.result and res.result.data_array:
            crops = [row[0] for row in res.result.data_array]
            return {c: {'t_base': 10.0, 'crop_type': 'fresh_produce'} for c in crops}
    except:
        pass
    return {}

CROP_DB = load_crops()

def get_coords(city):
    url = f"https://nominatim.openstreetmap.org/search?q={city}&format=json&limit=1"
    headers = {"User-Agent": "AgriLogistics_Internal_Tool_v2"}
    try:
        res = requests.get(url, headers=headers, timeout=10).json()
        return (float(res[0]['lat']), float(res[0]['lon'])) if res else None
    except:
        return None

# --- 3. WAYPOINT-BASED WEATHER ANALYSIS ---

def analyze_dynamic_path(route):
    """
    Analyzes route by sampling waypoints every 10km along the geometry.
    Fetches weather for each waypoint based on arrival time.
    Returns temperature stats and risk score using weighted formula.
    """
    coords = route['geometry']['coordinates']
    total_duration_sec = route['duration']
    total_distance_m = route['distance']
    sampling_interval_m = 10000  # 10km waypoint interval
    
    sampled_points = []
    accumulated_dist = 0
    
    # Sample waypoints every 10km along route
    for i in range(1, len(coords)):
        p1, p2 = (coords[i-1][1], coords[i-1][0]), (coords[i][1], coords[i][0])
        dist = geodesic(p1, p2).meters
        accumulated_dist += dist
        
        if accumulated_dist >= sampling_interval_m or i == len(coords)-1:
            progress_ratio = accumulated_dist / total_distance_m
            arrival_offset_hrs = int((progress_ratio * total_duration_sec) / 3600)
            sampled_points.append({
                "lat": p2[0], "lon": p2[1], 
                "hour_offset": min(arrival_offset_hrs, 23)
            })
            accumulated_dist = 0
    
    # Batch fetch weather for all waypoints
    lats = ",".join([str(p['lat']) for p in sampled_points])
    lons = ",".join([str(p['lon']) for p in sampled_points])
    w_url = f"https://api.open-meteo.com/v1/forecast?latitude={lats}&longitude={lons}&hourly=temperature_2m,relative_humidity_2m&timezone=Asia/Kolkata&forecast_days=1"
    
    try:
        w_res = requests.get(w_url, timeout=15).json()
        data_list = w_res if isinstance(w_res, list) else [w_res]
    except:
        # Fallback to default values if weather API fails
        return {
            "min_t": 25.0, "max_t": 25.0, "avg_t": 25.0,
            "avg_h": 60.0, "risk": 0.0,
            "time_hr": round(total_duration_sec / 3600, 2),
            "waypoint_count": len(sampled_points)
        }
    
    current_hour = datetime.now().hour
    
    temps, humids = [], []
    for idx, p in enumerate(sampled_points):
        if idx < len(data_list):
            target_hour = (current_hour + p['hour_offset']) % 24
            try:
                temps.append(data_list[idx]['hourly']['temperature_2m'][target_hour])
                humids.append(data_list[idx]['hourly']['relative_humidity_2m'][target_hour])
            except:
                temps.append(25.0)
                humids.append(60.0)
    
    # Calculate statistics
    avg_t = sum(temps) / len(temps) if temps else 25.0
    min_t = min(temps) if temps else 25.0
    max_t = max(temps) if temps else 25.0
    avg_h = sum(humids) / len(humids) if humids else 60.0
    dist_km = round(total_distance_m / 1000, 1)
    time_hr = round(total_duration_sec / 3600, 2)
    
    # Weighted spoilage score formula (from notebook)
    # Spoilage_Score = 0.3162*Max_Temp + 0.2195*Distance + 0.2140*Avg_Humidity + 0.1260*Min_Temp + 0.1243*Avg_Temp
    spoilage_score = (
        0.3162 * max_t +
        0.2195 * dist_km +
        0.2140 * avg_h +
        0.1260 * min_t +
        0.1243 * avg_t
    )
    
    return {
        "min_t": round(min_t, 1),
        "max_t": round(max_t, 1),
        "avg_t": round(avg_t, 1),
        "avg_h": round(avg_h, 1),
        "risk": round(spoilage_score, 2),
        "time_hr": time_hr,
        "waypoint_count": len(sampled_points)
    }

# --- 4. SARVAM AI FUNCTIONS ---

def routes_to_json(routes, origin, dest, crop_name):
    """Convert sim_routes list to structured JSON for LLM input."""
    route_data = []
    for r in routes:
        route_data.append({
            "route_name": f"Route {r['route_id']}",
            "distance_km": r['km'],
            "duration_hr": r['hrs'],
            "avg_temp": r['temp'],
            "max_temp": r['max_t'],
            "min_temp": r['min_t'],
            "avg_humidity": r['hum'],
            "risk_score": r['risk'],
            "road_condition": "standard",      # OSRM does not provide this; keep as default
            "overall_cost_value": r['risk'],   # risk score doubles as cost proxy for ranking
        })
    return {
        "origin": origin,
        "destination": dest,
        "crop": crop_name,
        "routes": route_data
    }

def build_prompts(payload):
    """Build system + user prompts from JSON payload."""
    routes    = payload["routes"]
    origin    = payload["origin"]
    dest      = payload["destination"]
    crop      = payload["crop"]

    df        = pd.DataFrame(routes)
    best_row  = df.loc[df["overall_cost_value"].idxmin()]
    worst_row = df.loc[df["overall_cost_value"].idxmax()]
    saving    = round(worst_row["overall_cost_value"] - best_row["overall_cost_value"], 2)

    spoilage_risk = df[df["avg_temp"] > 30]["route_name"].tolist()
    bruising_risk = df[df["road_condition"].str.lower().str.contains("poor")]["route_name"].tolist()

    def describe_all(df):
        lines = []
        for _, r in df.iterrows():
            line = (
                f"{r['route_name']} spans {r['distance_km']} kilometres, "
                f"takes {r['duration_hr']} hours, "
                f"average temperature {r['avg_temp']} degrees Celsius, "
                f"humidity {r['avg_humidity']} percent, "
                f"risk score {r['risk_score']}, "
                f"road condition is {r['road_condition']}, "
                f"and costs {r['overall_cost_value']} units"
            )
            lines.append(line)
        return "; and ".join(lines)

    all_routes_description = describe_all(df)

    SYSTEM_PROMPT = """You are an agricultural logistics route analysis agent speaking directly to a transport manager.
You have just finished analysing all available routes for a shipment.
You must speak in first person, as if you have personally evaluated every path and arrived at a conclusion.

Rules you must follow without exception:
- You must begin by saying that you have analysed all the paths.
- Then briefly mention what each path offers in terms of temperature, distance, and risk score.
- Then clearly state which single path you have found to be the most optimal, and give its temperature, distance, and risk score as the reason.
- Do not use bullet points, numbered lists, headers, or any formatting symbols.
- Do not use abbreviations. Write kilometre not km, degrees Celsius not C, percent not %.
- Do not use special characters like asterisks, dashes as separators, or colons mid-sentence.
- Write as one continuous spoken paragraph, naturally flowing, as if being read aloud by a voice assistant.
- Base your conclusion strictly on the data provided. Do not invent any facts.
- Do not output any thinking, reasoning, or working. Begin the paragraph directly."""

    USER_PROMPT = f"""You are analysing routes for a {crop} shipment from {origin} to {dest}.

Here is what each route offers:
{all_routes_description}.

Computed facts you must use exactly as given:
- The most optimal route based on lowest risk is {best_row['route_name']}.
- Its distance is {best_row['distance_km']} kilometres.
- Its duration is {best_row['duration_hr']} hours.
- Its average temperature is {best_row['avg_temp']} degrees Celsius.
- Its road condition is {best_row['road_condition']}.
- Its risk score is {best_row['risk_score']}.
- It saves {saving} risk units compared to the worst route.
- Routes with temperature above 30 degrees Celsius: {", ".join(spoilage_risk) if spoilage_risk else "none"}.
- Routes with poor road condition: {", ".join(bruising_risk) if bruising_risk else "none"}.

Speak as the agent. Start with the phrase that you have analysed all the paths. Briefly describe what each path offers. Then conclude by naming {best_row['route_name']} as the most optimal path and explain why using its temperature, distance, and risk. One flowing paragraph. Do not write any thinking or reasoning before the paragraph."""

    return SYSTEM_PROMPT, USER_PROMPT, best_row['route_name']

def call_sarvam_llm(system_prompt, user_prompt):
    response = llm_client.chat.completions.create(
        model="sarvam-m",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=800,
    )
    text = response.choices[0].message.content
    text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL)
    text = re.sub(r"[*_#`]", "", text)
    text = re.sub(r"\n+", " ", text)
    text = re.sub(r"\s{2,}", " ", text)
    return text.strip()

def translate_text(text, target_lang):
    def translate_chunks(text, target_lang, source_lang="en-IN", max_chars=900):
        sentences = re.split(r'(?<=[.!?])\s+', text)
        chunks, current = [], ""
        for s in sentences:
            if len(current) + len(s) + 1 <= max_chars:
                current += (" " if current else "") + s
            else:
                if current: chunks.append(current)
                current = s
        if current: chunks.append(current)
        if not chunks: chunks = [text[:max_chars]]
        results = []
        for chunk in chunks:
            r = requests.post(
                "https://api.sarvam.ai/translate",
                headers=SARVAM_HEADERS,
                json={
                    "input": chunk,
                    "source_language_code": source_lang,
                    "target_language_code": target_lang,
                    "speaker_gender": "Male",
                    "mode": "formal",
                    "enable_preprocessing": True,
                }
            )
            r.raise_for_status()
            results.append(r.json()["translated_text"])
        return " ".join(results)
    return translate_chunks(text, target_lang)

def tts_rest(text, lang, speaker="shubh", pace=1.0):
    MAX_CHARS = 450
    sentences = re.split(r'(?<=[।.!?])\s+', text)
    chunks, current = [], ""
    for s in sentences:
        if len(s) > MAX_CHARS:
            sub_parts = re.split(r'(?<=,)\s+', s)
            for part in sub_parts:
                if len(current) + len(part) + 1 <= MAX_CHARS:
                    current += (" " if current else "") + part
                else:
                    if current: chunks.append(current)
                    current = part[:MAX_CHARS]
        elif len(current) + len(s) + 1 <= MAX_CHARS:
            current += (" " if current else "") + s
        else:
            if current: chunks.append(current)
            current = s
    if current: chunks.append(current)
    if not chunks: chunks = [text[:MAX_CHARS]]

    all_audio = b""
    for chunk in chunks:
        payload = {
            "inputs": [chunk],
            "target_language_code": lang,
            "speaker": speaker,
            "model": "bulbul:v3",
            "pace": pace,
            "speech_sample_rate": 22050,
            "enable_preprocessing": True,
        }
        r = requests.post(
            "https://api.sarvam.ai/text-to-speech",
            headers=SARVAM_HEADERS,
            json=payload,
            timeout=60
        )
        r.raise_for_status()
        all_audio += base64.b64decode(r.json()["audios"][0])
    return all_audio

# --- 5. UI: INPUT CONTROL PANEL ---
st.title("🚛 KrishiMarg: Spatio-Temporal Risk Portal")
st.markdown("Multi-route optimization using waypoint-based weather sampling (10km intervals).")

with st.sidebar:
    st.header("🕹️ Simulation Controls")
    origin = st.text_input("Origin City", "Pune")
    dest   = st.text_input("Destination City", "Chennai")
    if CROP_DB:
        crop_name = st.selectbox("Select Commodity", sorted(CROP_DB.keys()))
        st.caption(f"📦 {len(CROP_DB)} crops available in database")
        if crop_name in CROP_DB:
            with st.expander("🌿 Crop Profile"):
                for k, v in CROP_DB[crop_name].items():
                    st.write(f"**{k.replace('_', ' ').title()}:** {v}")
    else:
        crop_name = st.text_input("Crop", "Tomato")
        st.warning("⚠️ Crop database unavailable. Type crop name manually.")

    run_btn = st.button("🚀 Run Analysis")

    st.markdown("---")
    st.header("🤖 AI Recommendation")
    target_lang = st.selectbox(
        "Output Language",
        ["en-IN", "hi-IN", "ta-IN", "te-IN", "kn-IN", "mr-IN", "bn-IN", "gu-IN"],
        index=1,
        help="Language for LLM recommendation & voice output"
    )
    #speaker = st.selectbox("Voice Speaker", ["shubh", "meera", "arjun", "diya"], index=0)
    pace    = st.slider("Speech Pace", 0.5, 2.0, 1.0, 0.1)

    ai_btn = st.button(
        "🎙️ Generate AI Recommendation",
        disabled=(st.session_state['current_sim'] is None),
        help="Run a route simulation first"
    )

    if st.button("🗑️ Clear All History"):
        w.statement_execution.execute_statement(
            warehouse_id=WAREHOUSE_ID,
            statement=f"TRUNCATE TABLE {ANALYTICS_TABLE}",
            wait_timeout="30s"
        )
        st.session_state['current_sim']      = None
        st.session_state['ai_recommendation'] = None
        st.cache_data.clear()
        st.rerun()

# --- 6. ENGINE: WAYPOINT-BASED ROUTE ANALYSIS ---
if run_btn:
    with st.spinner(f"Computing spatio-temporal risk for {crop_name}..."):
        p1, p2 = get_coords(origin), get_coords(dest)
        if not p1 and not p2:
            st.error(f"❌ Could not find both '{origin}' and '{dest}'.")
        elif not p1:
            st.error(f"❌ Could not find '{origin}'.")
        elif not p2:
            st.error(f"❌ Could not find '{dest}'.")
        elif not CROP_DB:
            st.error("❌ Crop database could not be loaded.")
        elif crop_name.lower() not in [k.lower() for k in CROP_DB.keys()]:
            st.error(f"❌ '{crop_name}' not in crop database.")
        else:
            # Fetch alternative routes from OSRM
            osrm = requests.get(
                f"https://router.project-osrm.org/route/v1/driving/{p1[1]},{p1[0]};{p2[1]},{p2[0]}?overview=full&geometries=geojson&alternatives=true",
                timeout=15
            ).json()

            if 'routes' in osrm and osrm['routes']:
                route_colors = ['#e74c3c', '#2980b9', '#27ae60', '#f39c12']
                sim_routes   = []

                for i, route in enumerate(osrm['routes']):
                    # Use waypoint-based analysis from notebook
                    stats = analyze_dynamic_path(route)
                    
                    km   = round(route['distance'] / 1000, 1)
                    geom = route['geometry']

                    sim_routes.append({
                        "route_id": i + 1, 
                        "color": route_colors[i % len(route_colors)],
                        "risk": stats['risk'], 
                        "hrs": stats['time_hr'], 
                        "km": km,
                        "temp": stats['avg_t'], 
                        "max_t": stats['max_t'],
                        "min_t": stats['min_t'], 
                        "hum": stats['avg_h'], 
                        "geom": geom,
                        "waypoint_count": stats['waypoint_count']
                    })

                    # Store in analytics table
                    sql = f"""INSERT INTO {ANALYTICS_TABLE}
                              (crop, route_id, dist_km, time_hr, avg_temp, max_temp, min_temp, avg_humidity, risk_score, runtime, geometry)
                              VALUES ('{crop_name}', {i+1}, {km}, {stats['time_hr']}, {stats['avg_t']}, {stats['max_t']}, {stats['min_t']}, {stats['avg_h']}, {stats['risk']}, current_timestamp(), '{json.dumps(geom)}')"""
                    try:
                        w.statement_execution.execute_statement(
                            warehouse_id=WAREHOUSE_ID, statement=sql, wait_timeout="30s"
                        )
                    except Exception as e:
                        st.warning(f"Could not save route {i+1} to database: {e}")

                st.session_state['current_sim'] = {
                    "origin": origin, "dest": dest, "p1": p1,
                    "crop": crop_name, "routes": sim_routes
                }
                st.session_state['ai_recommendation'] = None   # reset old recommendation
                st.cache_data.clear()
                st.success(f"✅ Analyzed {len(sim_routes)} routes using waypoint-based weather sampling!")
            else:
                st.error("❌ Could not fetch routes from OSRM. Check network connectivity.")

# --- 7. AI RECOMMENDATION ENGINE ---
if ai_btn and st.session_state['current_sim']:
    res = st.session_state['current_sim']
    with st.spinner("🤖 Sarvam AI is analysing routes..."):
        try:
            # Step 1: Build structured JSON from sim results
            payload = routes_to_json(res['routes'], res['origin'], res['dest'], res['crop'])

            # Step 2: Generate LLM paragraph (sarvam-m)
            sys_prompt, user_prompt, best_route = build_prompts(payload)
            english_text = call_sarvam_llm(sys_prompt, user_prompt)

            # Step 3: Translate if needed
            if target_lang == "en-IN":
                final_text = english_text
            else:
                final_text = translate_text(english_text, target_lang)

            # Step 4: TTS (bulbul:v3)
            audio_bytes = tts_rest(final_text, target_lang, "shubh", pace)

            st.session_state['ai_recommendation'] = {
                "english": english_text,
                "translated": final_text,
                "audio": audio_bytes,
                "lang": target_lang,
                "best_route": best_route,
                "payload_json": payload
            }
        except Exception as e:
            st.error(f"❌ AI Recommendation failed: {e}")

# --- 8. DISPLAY AI RECOMMENDATION ---
if st.session_state['ai_recommendation']:
    rec = st.session_state['ai_recommendation']
    st.markdown("---")
    st.subheader("🤖 AI-Powered Route Recommendation")

    col1, col2 = st.columns([2, 1])
    with col1:
        if rec['lang'] != "en-IN":
            with st.expander("📝 English Original"):
                st.info(rec['english'])
            st.success(f"**{rec['lang']} Translation:**\n\n{rec['translated']}")
        else:
            st.success(rec['english'])

    with col2:
        st.metric("✅ Recommended Route", rec['best_route'])
        with st.expander("🗂️ Route JSON Payload"):
            st.json(rec['payload_json'])

    st.audio(rec['audio'], format="audio/wav")

# --- 9. VISUALIZATION 1: CURRENT ROUTE(S) ---
if st.session_state['current_sim']:
    res = st.session_state['current_sim']
    st.markdown("---")
    st.subheader(f"Current Analysis: {res['origin']} to {res['dest']}")

    cols = st.columns(len(res['routes']))
    for i, r in enumerate(res['routes']):
        cols[i].metric(
            f"Route {r['route_id']} Risk",
            f"{r['risk']} Score",
            f"{r['km']} km | {r['hrs']}h | {r.get('waypoint_count', 0)} waypoints"
        )

    m_curr = folium.Map(location=res['p1'], zoom_start=6, tiles="CartoDB Positron")
    for r in res['routes']:
        folium.GeoJson(
            r['geom'],
            style_function=lambda x, c=r['color']: {'color': c, 'weight': 6, 'opacity': 0.85},
            tooltip=f"Route {r['route_id']}: Risk {r['risk']} | {r['temp']}°C avg (min: {r['min_t']}°C, max: {r['max_t']}°C) | {r['hum']}% humidity | {r['hrs']}h | {r['km']}km"
        ).add_to(m_curr)
    st_folium(m_curr, width=1200, height=400, key="current_map")

    if len(res['routes']) > 1:
        st.markdown("### 📊 Route Comparison")
        best_risk_id = min(res['routes'], key=lambda r: r['risk'])['route_id']
        comparison = {
            "Metric": [
                "🏁 Distance (km)", "⏱️ Duration (hrs)", "🌡️ Avg Temp (°C)",
                "🔺 Max Temp (°C)", "🔻 Min Temp (°C)", "💧 Avg Humidity (%)",
                "⚠️ Spoilage Score", "📍 Waypoints Sampled", "✅ Recommendation",
            ]
        }
        for r in res['routes']:
            comparison[f"Route {r['route_id']}"] = [
                r['km'], r['hrs'], r['temp'], r['max_t'], r['min_t'], r['hum'], r['risk'],
                r.get('waypoint_count', 0),
                "✅ Preferred" if r['route_id'] == best_risk_id else "❌ Higher Risk",
            ]
        st.table(pd.DataFrame(comparison).set_index("Metric"))
